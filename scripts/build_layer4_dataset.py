"""Generate a strictly disjoint ChatML JSONL distillation dataset for Layer-4 small LLMs.

Produces verified (SYSTEM_PROMPT, symbolic_digest -> semantic_action JSON) training
and validation pairs targeting compact models like LiquidAI/LFM2.5-1.2B-Instruct.
Uses randomized generic entity IDs (never touching M1-M3 or Chambers 1-7 IDs) and
over-samples multi-stage Stage-2 button switching and fizzler/stall recovery.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portal_ml.planner import (
    SYSTEM_PROMPT,
    EpisodicMemory,
    RuleBasedStrategicPlanner,
    StrategicDirective,
    TriggerEvent,
    build_symbolic_digest,
    compile_semantic_action,
)
from portal_ml.schema.blackboard import (
    BlackboardSnapshot,
    EntityKind,
    EntityState,
    PlayerState,
    PortalWall,
    Vec3,
)

CUBE_PREFIXES = ("crate", "box", "weight", "block", "obj")
BTN_PREFIXES = ("pad", "plate", "switch", "floor_b", "btn")
PED_PREFIXES = ("ped", "pillar", "console", "timer_b")
EXIT_PREFIXES = ("door", "gate", "airlock", "exit")


def _build_synthetic_case(rng: random.Random, idx: int) -> dict[str, Any]:
    """Synthesize one randomized symbolic world state + exact semantic-action target."""
    scenario = rng.choices(
        [
            "multi_btn_stage1",
            "multi_btn_stage2",
            "fizzler_recovery",
            "stall_recovery",
            "compound_pedestal",
            "open_exit",
        ],
        weights=[16, 28, 18, 10, 20, 8],
        k=1,
    )[0]

    cp = rng.choice(CUBE_PREFIXES)
    bp = rng.choice(BTN_PREFIXES)
    pp = rng.choice(PED_PREFIXES)
    ep = rng.choice(EXIT_PREFIXES)
    n_zones = rng.randint(1, 3)

    entities: list[EntityState] = []
    mem = EpisodicMemory(max_entries=8)
    trigger = TriggerEvent.EPISODE_START
    prev_directive: StrategicDirective | None = None
    exit_id = f"{ep}_{idx % 9}"

    def rpos() -> Vec3:
        return Vec3(rng.uniform(48.0, 960.0), rng.uniform(48.0, 360.0), 0.0)

    if scenario == "open_exit":
        n_btns = rng.randint(0, 2)
        for k in range(n_btns):
            pos = rpos()
            z = rng.randrange(n_zones)
            entities.append(EntityState(id=f"{cp}_{k}", kind=EntityKind.CUBE, pos=pos, zone_id=z, active=True))
            entities.append(EntityState(id=f"{bp}_{k}", kind=EntityKind.FLOOR_BUTTON, pos=pos, zone_id=z, active=True))
        entities.append(EntityState(id=exit_id, kind=EntityKind.EXIT_DOOR, pos=rpos(), zone_id=0, active=True))
        trigger = TriggerEvent.EXIT_OPENED
        if n_btns > 0:
            prev_directive = StrategicDirective(subgoal=f"place {cp}_{n_btns - 1} on {bp}_{n_btns - 1}")
        mem.record(rng.randint(10, 80), "exit_opened", (exit_id,), f"exit door {exit_id} is now open")

    elif scenario == "multi_btn_stage1":
        n_pairs = rng.randint(2, 3)
        for k in range(n_pairs):
            entities.append(EntityState(id=f"{cp}_{k}", kind=EntityKind.CUBE, pos=rpos(), zone_id=rng.randrange(n_zones), active=False))
            entities.append(EntityState(id=f"{bp}_{k}", kind=EntityKind.FLOOR_BUTTON, pos=rpos(), zone_id=rng.randrange(n_zones), active=False))
        if rng.random() < 0.4:
            entities.append(EntityState(id=f"{pp}_0", kind=EntityKind.PEDESTAL_BUTTON, pos=rpos(), zone_id=0, active=False))
        entities.append(EntityState(id=exit_id, kind=EntityKind.EXIT_DOOR, pos=rpos(), zone_id=0, active=False))
        trigger = TriggerEvent.EPISODE_START

    elif scenario == "multi_btn_stage2":
        n_pairs = rng.randint(2, 3)
        n_done = rng.randint(1, n_pairs - 1)
        # Independently sample active cubes and pressed buttons so the model CANNOT
        # learn a spurious index shortcut (e.g. cube_0 -> btn_0, cube_1 -> btn_1).
        done_cube_indices = set(rng.sample(range(n_pairs), n_done))
        done_btn_indices = set(rng.sample(range(n_pairs), n_done))
        done_pairs = list(zip(sorted(done_cube_indices), sorted(done_btn_indices)))
        pair_pos: dict[int, tuple[Vec3, int]] = {}
        for ck, bk in done_pairs:
            pos = rpos()
            z = rng.randrange(n_zones)
            pair_pos[ck] = (pos, z)
            pair_pos[100 + bk] = (pos, z)

        for k in range(n_pairs):
            if k in done_cube_indices:
                pos, z = pair_pos[k]
                entities.append(EntityState(id=f"{cp}_{k}", kind=EntityKind.CUBE, pos=pos, zone_id=z, active=True))
            else:
                entities.append(EntityState(id=f"{cp}_{k}", kind=EntityKind.CUBE, pos=rpos(), zone_id=rng.randrange(n_zones), active=False))
        for k in range(n_pairs):
            if k in done_btn_indices:
                pos, z = pair_pos[100 + k]
                entities.append(EntityState(id=f"{bp}_{k}", kind=EntityKind.FLOOR_BUTTON, pos=pos, zone_id=z, active=True))
                mem.record(rng.randint(12, 70), "button_activated", (f"{bp}_{k}",), f"floor_button {bp}_{k} activated")
            else:
                entities.append(EntityState(id=f"{bp}_{k}", kind=EntityKind.FLOOR_BUTTON, pos=rpos(), zone_id=rng.randrange(n_zones), active=False))
        if rng.random() < 0.6:
            entities.append(EntityState(id=f"{pp}_0", kind=EntityKind.PEDESTAL_BUTTON, pos=rpos(), zone_id=0, active=False))
        entities.append(EntityState(id=exit_id, kind=EntityKind.EXIT_DOOR, pos=rpos(), zone_id=0, active=False))
        trigger = TriggerEvent.SUBGOAL_COMPLETED
        last_ck, last_bk = done_pairs[-1]
        prev_directive = StrategicDirective(subgoal=f"place {cp}_{last_ck} on {bp}_{last_bk}")

    elif scenario in ("fizzler_recovery", "stall_recovery"):
        n_cubes = rng.randint(2, 3)
        bad_idx = rng.randrange(n_cubes)
        for k in range(n_cubes):
            entities.append(EntityState(id=f"{cp}_{k}", kind=EntityKind.CUBE, pos=rpos(), zone_id=rng.randrange(n_zones), active=False))
        entities.append(EntityState(id=f"{bp}_0", kind=EntityKind.FLOOR_BUTTON, pos=rpos(), zone_id=0, active=False))
        entities.append(EntityState(id=exit_id, kind=EntityKind.EXIT_DOOR, pos=rpos(), zone_id=0, active=False))
        bad_id = f"{cp}_{bad_idx}"
        prev_directive = StrategicDirective(subgoal=f"place {bad_id} on {bp}_0")
        if scenario == "fizzler_recovery":
            trigger = TriggerEvent.CUBE_FIZZLED
            mem.record(rng.randint(10, 40), "cube_fizzled", (bad_id,), f"cube {bad_id} vaporized by fizzler; avoid reusing this cube")
        else:
            trigger = TriggerEvent.STALL_DETECTED
            mem.record(rng.randint(45, 90), "cube_stalled", (bad_id,), f"no progress while targeting {bad_id}")

    else:  # compound_pedestal (all floor buttons already pressed -> press pedestal)
        n_pairs = 0 if rng.random() < 0.15 else rng.randint(1, 3)
        perm_btns = list(range(n_pairs))
        rng.shuffle(perm_btns)
        for k in range(n_pairs):
            bk = perm_btns[k]
            z = rng.randrange(n_zones)
            pos = rpos()
            entities.append(EntityState(id=f"{cp}_{k}", kind=EntityKind.CUBE, pos=pos, zone_id=z, active=True))
            entities.append(EntityState(id=f"{bp}_{bk}", kind=EntityKind.FLOOR_BUTTON, pos=pos, zone_id=z, active=True))
            mem.record(rng.randint(15, 80), "button_activated", (f"{bp}_{bk}",), f"floor_button {bp}_{bk} activated")
        entities.append(EntityState(id=f"{pp}_0", kind=EntityKind.PEDESTAL_BUTTON, pos=rpos(), zone_id=0, active=False))
        entities.append(EntityState(id=exit_id, kind=EntityKind.EXIT_DOOR, pos=rpos(), zone_id=0, active=False))
        if n_pairs > 0:
            last_ck = n_pairs - 1
            last_bk = perm_btns[-1]
            prev_directive = StrategicDirective(subgoal=f"place {cp}_{last_ck} on {bp}_{last_bk}")
            if rng.random() < 0.25:
                trigger = TriggerEvent.STALL_DETECTED
                mem.record(rng.randint(85, 130), "cube_stalled", (f"{cp}_{last_ck}",), f"no progress for 45 ticks while targeting {cp}_{last_ck}")
            else:
                trigger = TriggerEvent.SUBGOAL_COMPLETED
        else:
            trigger = TriggerEvent.EPISODE_START

    walls = [
        PortalWall(id=f"wall_z{z}", pos=Vec3(128.0 + z * 320.0, 0.0, 64.0), normal=Vec3(0.0, 1.0, 0.0), zone_id=z)
        for z in range(n_zones)
    ]
    snap = BlackboardSnapshot(
        tick=rng.randint(0, 120),
        dt=0.1,
        chamber_id=900 + (idx % 50),
        player=PlayerState(pos=rpos(), zone_id=rng.randrange(n_zones)),
        entities=entities,
        portal_walls=walls,
    )

    # Derive ground-truth semantic action via deterministic RuleBasedStrategicPlanner
    oracle = RuleBasedStrategicPlanner()
    directive = oracle.plan(snap, mem, trigger)
    ov = directive.role_overrides
    if any(e.active for e in snap.entities_of_kind(EntityKind.EXIT_DOOR)):
        target_payload = {
            "action": "go_to_exit",
            "cube_id": None,
            "button_id": None,
            "pedestal_id": None,
            "exit_id": ov.get("exit"),
        }
    elif ov.get("floor_button") is not None and ov.get("cube") is not None:
        target_payload = {
            "action": "place_cube_on_button",
            "cube_id": ov["cube"],
            "button_id": ov["floor_button"],
            "pedestal_id": None,
            "exit_id": ov.get("exit"),
        }
    elif ov.get("pedestal_button") is not None:
        target_payload = {
            "action": "press_pedestal",
            "cube_id": None,
            "button_id": None,
            "pedestal_id": ov["pedestal_button"],
            "exit_id": ov.get("exit"),
        }
    else:
        target_payload = {
            "action": "go_to_exit",
            "cube_id": None,
            "button_id": None,
            "pedestal_id": None,
            "exit_id": ov.get("exit"),
        }

    # Verify target compiles cleanly with zero warnings against the live snapshot
    compiled = compile_semantic_action(snap, **target_payload)
    _, warnings = compiled.validate_against_snapshot(snap)
    assert not warnings, f"Synthetic dataset validation warnings: {warnings}"

    digest = build_symbolic_digest(snap, mem, trigger, prev_directive)
    return {
        "scenario": scenario,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": digest},
            {"role": "assistant", "content": json.dumps(target_payload, separators=(",", ":"))},
        ],
    }


def generate_dataset(count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return [_build_synthetic_case(rng, i) for i in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Layer-4 distillation JSONL datasets.")
    parser.add_argument("--train-count", type=int, default=500)
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-out", default="records/layer4_distill_train.jsonl")
    parser.add_argument("--val-out", default="records/layer4_distill_val.jsonl")
    args = parser.parse_args()

    train_rows = generate_dataset(args.train_count, args.seed)
    val_rows = generate_dataset(args.val_count, args.seed + 10_000)

    for path_str, rows in ((args.train_out, train_rows), (args.val_out, val_rows)):
        p = Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fp:
            for r in rows:
                fp.write(json.dumps({"messages": r["messages"]}, separators=(",", ":")) + "\n")
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["scenario"]] = counts.get(r["scenario"], 0) + 1
        print(f"Wrote {len(rows)} examples -> {p} | breakdown={counts}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
