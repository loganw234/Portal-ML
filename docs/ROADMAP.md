# Portal-ML Roadmap

Started 2026-11-26. Objective: prove a decoupled 3-layer neuroevolutionary reasoning agent in *Portal* (Source Engine) — readable Behavior Tree DSL for causal reasoning over a symbolic Blackboard, decoupled from swappable low-level motor primitives.

## Phase 1 — Decoupled Reasoning PoC & Source Engine Bridge (Done, 2026-11-26)

- Layer 1 symbolic Blackboard schema (`portal_ml/schema/blackboard.py`) in Source Hammer units.
- Layer 2 Behavior Tree DSL (`SELECTOR`, `SEQUENCE`, `CONDITION`, `ACTION`) parser, pretty-printer, and tick evaluator (`portal_ml/bt/`).
- Layer 3 deterministic motor primitive interface (`MoveTo`, `Grab`, `Drop`, `Press`, `ShootPortal`) with Phase 2 `MotorPolicy` protocol (`portal_ml/motor/`).
- 5-chamber mock 3D simulator (`portal_ml/sim/`) at ~1,978 episodes/sec (42.7k ticks/sec).
- Evolutionary BT engine (`portal_ml/evolution/`) achieving 4/4 train solve on 4/5 seeds and 2/5 zero-shot pass on held-out Chamber 5 (`decoupled`), vs 0/5 train and 0/5 Chamber 5 (`unconstrained`).
- Programmatic single-map 5-chamber `.vmf` generator (`portal_ml/vmf/`) compiled cleanly through Portal 1 `vbsp`/`vvis`/`vrad` with zero leaks.
- 32-bit Win32 C++17 server plugin (`source_plugin/`, `ISERVERPLUGINCALLBACKS003`) compiled cleanly with MSVC, installed into Portal 1, and verified live over TCP IPC (`127.0.0.1:27500`).

---

## Completed Milestones

### Milestone 1 — Sim-to-Source Replay & Live Physics Calibration (Done, 2026-11-26)
- Saved mock-evolved champion `champions/seed7_decoupled.bt` (24 AST nodes, trained strictly on Chambers 1–4) and built `scripts/replay_live.py` with JSONL tick trace recording.
- Diagnosed and closed live Source Engine physics gaps across 6 iterative live Portal 1 runs:
  - Disabled joystick drift (`-nojoy`) and client console input blocking (removed `-console` default from `launch_portal.ps1`).
  - Calibrated solid-prop arrival in `portal_ml/bt/evaluator.py` (`42u` for solid cubes/pedestals matching 32u player+prop collision hulls; `28u` for walkable floor buttons).
  - Switched `MoveTo` to closed-loop server-side `basevelocity` delta impulses (once per 6-subtick step) with portal-crossing stop guard and two-stage perpendicular portal approach.
  - Thinned portal concrete panels to `1u` in `portal_ml/vmf/generator.py` (eliminating 8u corner snags) and added `point_teleport` reset targets (`spawnflags=1`) for clean in-place cube resets.
  - Held carried cubes overhead (`z+108`, `solid 0`) and dropped via server-side `EyeAngles` aim + `ent_teleport` + clear step + `DisableMotion` (honest limit: `CC_Ent_Teleport` places `prop_physics` center on the hit surface without lifting by half-height; direct `IPhysicsObject::SetPosition` is queued for Milestone 4).
- **Verified 5/5 live Portal 1 pass** (`records/live_seed7_2026-11-26.jsonl`): Ch1 11 steps, Ch2 27 steps, Ch3 20 steps, Ch4 11 steps, **held-out Ch5 38 steps zero-shot** (~104ms in-place resets across 6 settle ticks).

### Milestone 2 — Phase 2: Neuroevolved Neural Motor Primitives (Done, 2026-11-26)
- Added continuous body-frame control (`MoveVector(forward, strafe, yaw_delta)`) to `portal_ml/motor/primitives.py`, `portal_ml/sim/mock_env.py`, and the 32-bit C++ Source Engine plugin (`source_plugin/src/motor_controller.cpp`, `portal_ml_plugin.cpp`).
- Built pure-Python inspectable `MLP` with strict schema-checked checkpoints (`CHECKPOINT-INPUTS` / `CHECKPOINT-OUTPUTS` / `CHECKPOINT-WEIGHTS`), `NeuralMoveToPolicy` (`6->8->3`, 83 weights), `NeuralAimPolicy` (`4->6->2`, 44 weights), `HybridMotorController`, and `portal_ml/evolution/motor_trainer.py`.
- Neuroevolved both networks from random weights on chamber-agnostic synthetic egocentric tasks (`scripts/train_phase2_motor.py --seed 7`: 100.0% validation on both).
- Verified progressive primitive swaps underneath the **frozen** `champions/seed7_decoupled.bt` (zero changes to Layer 1 or Layer 2):
  - **Phase 2A (neural `MoveTo`):** 5/5 mock (Ch1–5: 16, 18, 27, 10, 47) and **5/5 live Portal 1** (Ch1–5: 11, 12, 23, 10, 51; `records/live_phase2a_2026-11-26.jsonl`).
  - **Phase 2B (neural `MoveTo` + neural `ShootPortal Aim`):** 5/5 mock (Ch1–5: 16, 18, 27, 20, 67) and **5/5 live Portal 1** (Ch1–5: 12, 12, 24, 24, 85; `records/live_phase2b_2026-11-26.jsonl`).

### Milestone 3 — Generalization Hardening, Expanded Held-Out Suite & Real-Game Audit (Done, 2026-11-26)
- Added two harder held-out chambers to `portal_ml/sim/chambers.py`, `source_plugin/src/blackboard_extractor.cpp`, and `maps/portal_ml_5chambers.vmf` (956 triangles, zero leaks):
  - **Chamber 6 (`Held-Out: Button Across Gap`):** Opposite asymmetry from Chamber 5 (cube + closed exit at spawn; floor button across pit; opposite-facing portal walls).
  - **Chamber 7 (`Held-Out: Three-Zone Relay`):** Three islands separated by two pits (exit on zone 0, button on zone 1, cube on zone 2), requiring two mid-episode portal re-links (`0->2`, `2->1`, `1->0`).
- Diagnosed why earlier 4/4 decoupled champions failed held-out rooms and implemented two zero-leakage fixes:
  1. `Drop()` target-locality symmetry in `portal_ml/bt/evaluator.py` (matching `Grab` and `Press` preconditions when a target entity is bound).
  2. Training-only causal invariance augmentations strictly on Training Chambers 3 & 4 (`ch3_distractor_walls` and `ch4_stale_portals` in `MockPortalEnv` / `fitness.py`), breaking training-set spurious correlations without touching held-out Chambers 5–7.
- Measured 10-seed progression (`[1, 2, 3, 5, 7, 11, 13, 17, 21, 42]`, pop 120, gens 120): baseline `3/10` -> `+ Drop()` symmetry `5/10` -> `+ causal augmentation` **`7/10` passing all three held-out chambers (5, 6, 7)** (Ch5: 7/10, Ch6: 8/10, Ch7: 7/10). Canonical 5-seed ablation: `decoupled` **4/5 (80%)** vs `unconstrained` **1/5 (20%)** across all three held-out chambers.
- Verified **7/7 chambers PASS live in Portal 1** for BOTH Phase 1 scripted (`11/27/19/11/39/39/47` steps, 1.50s; `records/live_7ch_phase1_2026-11-26.jsonl`) and Phase 2B neural `MoveTo(83w) + Aim(44w)` (`11/10/20/22/61/72/77` steps, 2.23s; `records/live_7ch_phase2b_2026-11-26.jsonl`) after fixing post-drop clear step (`72u` off the 80x80 button) and server-side yaw tracking (`tracked_yaw_` + `ServerCommand setang`).
- Audited all 24 real Portal 1 campaign `.bsp` maps in `portal_pak_dir.vpk` (`scripts/audit_campaign_maps.py`) and documented the real-game feasibility and 4-upgrade full-game roadmap in [docs/REAL_GAME_ANALYSIS.md](REAL_GAME_ANALYSIS.md).

---

## Upcoming Milestones (Worked One at a Time)

### Milestone 4 — Native Source Control & Generic Live-BSP Discovery
- Upgrade `motor_controller.cpp` from console commands (`ent_teleport` / `basevelocity`) to direct `CUserCmd` + `IPhysicsObject::SetPosition` control and live milestone reward reporting in C++.
- Replace static `kLayouts` in `blackboard_extractor.cpp` with dynamic server-edict & BSP portalable-surface discovery (`LevelInit` parser) so the bridge runs on arbitrary maps without hardcoded coordinates.

### Milestone 5 — First Real Campaign Chamber Benchmark (`testchmb_a_02`)
- Load an unmodified Valve Portal 1 campaign map (`map testchmb_a_02`) through the generic Blackboard extractor, add fizzler awareness, and benchmark live real-game execution (see [docs/REAL_GAME_ANALYSIS.md](REAL_GAME_ANALYSIS.md)).
