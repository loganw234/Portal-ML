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

## Upcoming Milestones (Worked One at a Time)

### Milestone 1 — Sim-to-Source Replay & Live Physics Calibration
- Evolve and save a champion BT DSL on `MockPortalEnv`, then replay it end-to-end inside live Portal 1 across Chambers 1–5.
- Record and close any sim-to-Source timing or physics discrepancies (`+use` pickup alignment, `func_weight_button` settling, portal entry/exit momentum, `func_door` open latency) so a mock-evolved champion clears Chambers 1–5 live in Portal 1.

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
