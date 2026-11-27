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

---

## Upcoming Milestones (Worked One at a Time)

### Milestone 2 — Phase 2: Neuroevolved Neural Motor Primitives
- Replace Phase-1 deterministic primitives one by one with small neuroevolved neural networks implementing `MotorPolicy` (`forward`, `strafe`, `yaw_delta`, `pitch_delta`, `use`, `fire_blue`, `fire_orange`):
  1. Swap `MoveTo` first under a **frozen evolved BT** and verify Chambers 1–5 still pass.
  2. Swap `ShootPortal` (aiming onto portalable surfaces).
  3. Swap `Grab` / `Drop` / `Press`.

### Milestone 3 — Generalization Hardening & Expanded Held-Out Suite
- Diagnose why some 4/4 decoupled champions fail Chamber 5 (portal overwrite order on return crossings) and improve zero-shot generalization consistency across seeds.
- Add further held-out test chambers (e.g., Chamber 6: floor button across pit with cube at spawn; Chamber 7: pedestal + cube + pit 3-stage puzzle).

### Milestone 4 — Native Source Control & High-Speed Headless Stepping
- Upgrade `motor_controller.cpp` from console command strings (`setang` / `+forward`) to direct `CUserCmd` / physics velocity control and live milestone reward reporting in C++.
- Benchmark live Portal 1 throughput under fixed timesteps and fast timescale / `-textmode`.
