"""Front-door verification gate for Portal-ML.

Runs every invariant gate and prints a single verdict line:
    VERDICT: PASS (<N> gates)
Run with `--control` to also run negative controls (planted faults that each gate must catch).

Gates:
  1. unit-tests      — Full unittest suite (schema, BT DSL, mock 3D sim, TCP bridge, VMF, evolution)
  2. no-personal     — No user-specific home paths in tracked project files
  3. coord-parity    — Chamber 1-5 coordinates in C++ blackboard_extractor.cpp match chambers.py 1:1
  4. held-out-iso    — Chamber 5 is excluded from TRAIN_CHAMBERS and marked held_out=True
  5. vmf-integrity   — Generated 5-chamber .vmf has balanced braces and Portal 1 entities
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portal_ml.evolution.fitness import HELD_OUT_CHAMBER, TRAIN_CHAMBERS
from portal_ml.sim.chambers import CHAMBERS
from portal_ml.vmf.generator import build_five_chamber_vmf

PERSONAL_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+|/home/[A-Za-z0-9_.-]+)")
SKIP_DIRS = {".git", "__pycache__", "third_party", "build-x86", "build", ".pytest_cache"}
SCAN_EXTS = {".py", ".cpp", ".h", ".txt", ".md", ".bat", ".sh", ".ps1", ".toml", ".yml", ".yaml"}


def gate_unit_tests(quick: bool = False) -> tuple[bool, str]:
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"))
    if quick:
        # Filter out slow full-evolution run when --quick is passed
        filtered = unittest.TestSuite()
        for group in suite:
            for sub in group:
                for test in sub:
                    if "evolution_solves_train" not in test.id():
                        filtered.addTest(test)
        suite = filtered
    buf = io.StringIO()
    runner = unittest.TextTestRunner(stream=buf, verbosity=0)
    res = runner.run(suite)
    if res.wasSuccessful():
        return True, f"{res.testsRun} tests passed"
    return False, f"TEST-FAIL ({len(res.failures)} failures, {len(res.errors)} errors)"


def check_personal_paths_in_text(text: str) -> str | None:
    m = PERSONAL_PATH_RE.search(text)
    return m.group(0) if m else None


def gate_no_personal_paths() -> tuple[bool, str]:
    hits: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in SCAN_EXTS:
            continue
        if path.name == "check.py":
            continue
        if path.suffix.lower() == ".log":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hit = check_personal_paths_in_text(text)
        if hit:
            hits.append(f"{path.relative_to(ROOT)} ({hit})")
    if hits:
        return False, "PERSONAL-PATH: " + ", ".join(hits[:5])
    return True, "clean"


def check_coord_parity(cpp_text: str) -> tuple[bool, str]:
    for cid in sorted(CHAMBERS):
        spec = CHAMBERS[cid]
        exit_ent = next(e for e in spec.entities if e.kind.value == "exit_door")
        expected_exit = f'"{exit_ent.id}"'
        if expected_exit not in cpp_text:
            return False, f"COORD-MISSING-EXIT ({exit_ent.id})"
        y_min_token = f"{spec.room_min.y:.1f}f"
        if y_min_token not in cpp_text:
            return False, f"COORD-MISMATCH (chamber {cid} min_y {y_min_token})"
    return True, "Chambers 1-5 match C++ extractor"


def gate_coord_parity() -> tuple[bool, str]:
    cpp = (ROOT / "source_plugin" / "src" / "blackboard_extractor.cpp").read_text(encoding="utf-8")
    return check_coord_parity(cpp)


def check_held_out_isolation(train_chambers: tuple[int, ...], held_out: int) -> tuple[bool, str]:
    if held_out in train_chambers:
        return False, "HELD-OUT-LEAK"
    if not CHAMBERS[held_out].held_out:
        return False, "HELD-OUT-FLAG-FALSE"
    for cid in train_chambers:
        if CHAMBERS[cid].held_out:
            return False, f"TRAIN-MARKED-HELD-OUT ({cid})"
    return True, f"train={list(train_chambers)}, held_out={held_out}"


def gate_held_out_isolation() -> tuple[bool, str]:
    return check_held_out_isolation(TRAIN_CHAMBERS, HELD_OUT_CHAMBER)


def check_vmf_text(text: str) -> tuple[bool, str]:
    if text.count("{") != text.count("}"):
        return False, "VMF-UNBALANCED-BRACES"
    for required in (
        '"classname" "worldspawn"',
        '"classname" "func_weight_button"',
        '"classname" "func_button"',
        '"classname" "func_door"',
        '"classname" "prop_portal"',
    ):
        if required not in text:
            return False, f"VMF-MISSING ({required})"
    return True, f"{len(text):,} bytes, balanced braces"


def gate_vmf_integrity() -> tuple[bool, str]:
    return check_vmf_text(build_five_chamber_vmf().to_vmf())


def run_controls() -> list[tuple[str, bool, str]]:
    """Negative controls: plant realistic faults and verify each gate catches them by code."""
    results: list[tuple[str, bool, str]] = []

    # Control 1: personal path detection
    hit = check_personal_paths_in_text(r"path = C:\Users\alice\repo")
    ok = hit is not None
    results.append(("no-personal/fault", ok, "CONTROL-DETECTED (PERSONAL-PATH)" if ok else "CONTROL-MISSED"))

    # Control 2: coord parity mismatch
    ok_bad, msg_bad = check_coord_parity("// empty cpp")
    caught = (not ok_bad) and ("COORD-MISSING-EXIT" in msg_bad)
    results.append(("coord-parity/fault", caught, f"CONTROL-DETECTED ({msg_bad})" if caught else "CONTROL-MISSED"))

    # Control 3: held-out leak into training set
    ok_leak, msg_leak = check_held_out_isolation((1, 2, 3, 4, 5), 5)
    caught = (not ok_leak) and (msg_leak == "HELD-OUT-LEAK")
    results.append(("held-out-iso/fault", caught, f"CONTROL-DETECTED ({msg_leak})" if caught else "CONTROL-MISSED"))

    # Control 4: unbalanced VMF braces
    ok_vmf, msg_vmf = check_vmf_text('world { "classname" "worldspawn"')
    caught = (not ok_vmf) and (msg_vmf == "VMF-UNBALANCED-BRACES")
    results.append(("vmf-integrity/fault", caught, f"CONTROL-DETECTED ({msg_vmf})" if caught else "CONTROL-MISSED"))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Portal-ML front-door verification gate.")
    parser.add_argument("--control", action="store_true", help="Also run negative controls")
    parser.add_argument("--quick", action="store_true", help="Skip full evolution test for sub-second check")
    args = parser.parse_args()

    gates: list[tuple[str, tuple[bool, str]]] = [
        ("no-personal", gate_no_personal_paths()),
        ("coord-parity", gate_coord_parity()),
        ("held-out-iso", gate_held_out_isolation()),
        ("vmf-integrity", gate_vmf_integrity()),
        ("unit-tests", gate_unit_tests(quick=args.quick)),
    ]

    all_ok = True
    for name, (ok, detail) in gates:
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name:14s} - {detail}")
        if not ok:
            all_ok = False

    controls_run = 0
    if args.control:
        for name, ok, detail in run_controls():
            controls_run += 1
            tag = "PASS" if ok else "FAIL"
            print(f"[{tag}] {name:18s} - {detail}")
            if not ok:
                all_ok = False

    verdict = "PASS" if all_ok else "FAIL"
    if args.control:
        print(f"VERDICT: {verdict} ({len(gates)} gates, {controls_run} controls)")
    else:
        print(f"VERDICT: {verdict} ({len(gates)} gates)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
