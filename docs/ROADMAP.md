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

### Milestone 4 — Native SendProp Physics, Live C++ Milestones & Generic BSP Discovery (Done, 2026-11-26)
- Upgraded `motor_controller.cpp` + `blackboard_extractor.cpp` from per-tick `ent_fire !player AddOutput basevelocity` console commands to direct SendProp `m_vecBaseVelocity` + Portal 1 `FL_BASEVELOCITY` (`1<<23`, compiled with `-DPORTAL=1` + SDK `const.h`) actuation (calibrated `0.72x` to match 240 u/s mock speed).
- Added live C++ milestone & reward computation (`ComputeStepReward` in `portal_ml_plugin.cpp`) matching `MockPortalEnv` 1:1 (`cube_grabbed` +10, `button_pressed` +25, `portal_*` +5, `portals_bridged` +15, `portal_traversed` +20, `exit_reached` +100).
- Verified **7/7 live Portal 1 PASS** with native velocity + live milestones for BOTH Phase 1 scripted (`13/35/22/11/51/40/43` steps, 2.66s; `records/live_m4_native_phase1_2026-11-26.jsonl`) and Phase 2B neural `MoveTo(83w) + Aim(44w)` (`12/58/21/21/69/62/98` steps, 2.79s, ~122 steps/sec, ~8.1ms resets; `records/live_m4_native_phase2b_2026-11-26.jsonl`).
- Built pure C++17 `BspScanner` (`source_plugin/src/bsp_scanner.cpp`) and matching Python `portal_ml/vmf/bsp_reader.py` parsing BSP v20 lumps (entities, planes, vertexes, texdata, texinfo, faces, edges, surfedges) automatically at `LevelInit` for arbitrary maps (`"op":"map_info"` IPC query).
- Verified live on Valve's unmodified campaign map **`testchmb_a_02`** via `scripts/inspect_live_map.py` (`records/live_testchmb_a02_2026-11-26.json`): auto-discovered 9 gameplay entities and 293 portalable concrete surfaces (walls, floors, ceilings) and moved the player cleanly via native `MoveVector`.
- Documented the proposed Layer-4 small-LLM strategic memory & subgoal planner in [docs/FOUR_LAYER_ARCHITECTURE.md](FOUR_LAYER_ARCHITECTURE.md).

### Milestone 5 — First Real Campaign Chamber Benchmark (`testchmb_a_02`) (Done, 2026-11-26)
- Extended `BspScanner` (`source_plugin/src/bsp_scanner.cpp`) to parse `weapon_portalgun` capabilities (`CanFirePortal1`/`CanFirePortal2`), `func_tracktrain` elevators, `trigger_portal_cleanser` fizzlers, and deduplicated paired sliding exit doors.
- Extended `BlackboardExtractor::ExtractGeneric` with vertical elevation zone clustering (`zone 0` main floor vs `zone 1` sunken pit), ceiling-dropper cube handling, live `prop_portal` outward normals, physical `func_door` displacement confirmation (`d > 52u`), and exit-to-elevator progression.
- Extended `MotorController` with native `weapon_portalgun` firing (`ent_fire weapon_portalgun FirePortal1`), 2-stage obstacle wall-following (clearing glass partitions and U-shaped button alcoves), pit-edge avoidance, and live-VPhysics `point_teleport` button drops respecting Valve's `floor_button_trigger_player_*` -> `floor_button_trigger_box_*` handoff timing.
- Added single-portal-gun remapping (`allow_orange_portal=False`) in `portal_ml/bt/evaluator.py` and automatic generic-map detection in `scripts/replay_live.py`.
- **Verified live zero-shot solve on unmodified Valve `testchmb_a_02` Room 1 (Test Chamber 04)** using the frozen 24-node `champions/seed7_decoupled.bt` (trained strictly on synthetic mock Chambers 1–4): fastest verified solve **106 steps (11.87s, reward 174.70)**; saved pass trace `records/live_testchmb_a02_solve_2026-11-26.jsonl` (**286 steps, 33.68s, reward 165.70**, all 7 milestones: `portal_orange_placed`, `cube_grabbed`, `portal_traversed`, `portal_blue_placed`, `portals_bridged`, `button_pressed`, `exit_reached`). Honest limit: multi-run drop repeatability within 350 steps is ~1/3–2/3 due to Valve's duplicate-`targetname` dropper cube + VPhysics rim rolling (native `+use` / `CPlayerPickupController` carry-and-drop queued for motor hardening).
- Re-verified **zero regressions on `portal_ml_5chambers`** (`-textmode`): **7/7 PASS** Phase 1 scripted (`13/35/22/11/49/39/45` steps) and **7/7 PASS** Phase 2B neural `MoveTo(83w) + Aim(44w)` (`13/59/21/20/72/61/86` steps).

### Milestone 6 — Layer-4 Event-Driven Strategic Memory & Subgoal Planner (Done, 2026-11-27)
- Extended `WorkingMemory` (`portal_ml/schema/blackboard.py`) and `resolve_entity` / `cube_on_button` (`portal_ml/bt/predicates.py`) with optional Layer-4 `role_overrides` (supporting both live entity bindings and explicit `None` role masks, while staying 100% bit-for-bit identical in 3-layer mode).
- Extended `MockPortalEnv` / `ChamberSpec` with `FizzlerBarrier` (`trigger_portal_cleanser` carried-cube vaporization emitting `cube_fizzled:<id>`) and `require_all_buttons` multi-button door logic, keeping `CHAMBERS[1..7]` and `HELD_OUT_CHAMBERS` completely untouched.
- Built zero-dependency `portal_ml/planner/` package: bounded deduplicated `EpisodicMemory(max_entries=8)`, `StrategicDirective.validate_against_snapshot()`, dynamic `build_directive_json_schema()`, compact `build_symbolic_digest()`, `NullStrategicPlanner`, `RuleBasedStrategicPlanner`, `OpenAICompatibleLLMPlanner` (LM Studio / Ollama / `llama-server` with `<think>...</think>` stripping, HTTP 400 retry, and timeout fallback), `FourLayerAgent`, and 3 multi-stage benchmark chambers (`M1: Dual-Button Relay`, `M2: Fizzler-Gated Decoy Cube`, `M3: Three-Stage Compound Vault`).
- Wired `--planner {none,rule,llm}` into `scripts/replay_live.py` and built `scripts/ablate_layer4.py`.
- **Verified controlled 3-layer vs 4-layer ablation (deterministic + live local LLMs via LM Studio, 2026-11-27) under frozen `champions/seed7_decoupled.bt` (24 nodes, zero retraining):**
  - **Rule baseline (`records/ablation_layer4_2026-11-27.json`):** 3-layer `7/7` standard + `0/3` multi-stage vs 4-layer (`rule`) **`7/7` standard + `3/3` multi-stage** (scripted: `69/66/94` steps; Phase 2B neural: `87/82/149` steps; `3/3/4` calls; 0 fallbacks/warnings).
  - **Live `phi-4-mini-instruct` (3.8B, 2.49 GB, `records/ablation_layer4_phi4mini_2026-11-27.json`):** **`7/7` standard + `3/3` multi-stage** on BOTH Phase 1 scripted (`116/66/91` steps, 23 total calls, avg 470 ms/call) AND Phase 2B neural (`81/82/152` steps, 22 total calls, avg 461 ms/call) — **20/20 zero-shot PASS**, 0 fallbacks, 0 warnings.
  - **Live `liquid/lfm2.5-1.2b` (1.2B, 1.25 GB, `records/ablation_layer4_lfm25_2026-11-27.json`):** **`7/7` standard + `1/3` multi-stage** on both scripted (`M2` PASS 42 steps / 2 calls; avg 368 ms/call) and Phase 2B neural (`M2` PASS 65 steps / 2 calls; avg 332 ms/call), 0 fallbacks/warnings — ~28% faster at half the VRAM, exposing a clean Stage-2 multi-button switching ceiling on M1/M3 for LoRA distillation.
- Verified `OpenAICompatibleLLMPlanner` end-to-end against an in-process loopback OpenAI HTTP server (including `<think>` stripping, dynamic `json_schema` verification, HTTP 400 fallback, and unreachable-server fallback) in `tests/test_layer4_planner.py`.

### Milestone 7 — Human-Equivalent Physical Connections & Neuroevolved Parkour/Hazard Skill (Done, 2026-11-27)
- **Layer-1 human-equivalent non-visual perception (Python + 32-bit C++ Source Engine parity):**
  - Extended `PlayerState` (`portal_ml/schema/blackboard.py`, `source_plugin/src/blackboard_extractor.{h,cpp}`) with live proprioception (`health`, `on_ground`, `ducking`, `water_level`) read natively from Source SendProps (`m_iHealth`, `m_fFlags` `FL_ONGROUND`/`FL_DUCKING`, `m_nWaterLevel`).
  - Added 13-channel egocentric `LocalSensors` ("non-visual LiDAR/touch": 8 chest-height horizontal `wall_rays`, 3 forward `floor_probes`, knee-vs-chest `step_clearance`, and `hazard_ahead`) computed analytically in `MockPortalEnv` and natively via **`IEngineTrace` (`EngineTraceServer003`, `SkipPlayerFilter`)** in `source_plugin` (zero extra `.lib` dependencies).
  - Added `HazardVolume` (`trigger_hurt` / acid goo) + `ObstacleBox` (solid tall walls & jumpable hurdles) to `MockPortalEnv` and BSP Lump 14 (`LUMP_MODELS`) `trigger_hurt` AABB discovery to `BspScanner`.
- **Layer-3 full human body actuators (Python + C++ parity):**
  - Extended `MoveVector` (`portal_ml/motor/primitives.py`, `MockPortalEnv`, `motor_controller.{h,cpp}`) to full 6-DOF body + interaction control: `forward`, `strafe`, `yaw_delta`, `pitch_delta`, `jump`, `crouch`, `use`, `fire_blue`, `fire_orange` — backward-compatible with all Phase-2 checkpoints.
  - Added 3D jump/gravity physics (`sv_gravity 600`, `JUMP_SPEED 240`, hurdle clearance, cross-pit gap jumping, acid trench jumping) in `MockPortalEnv` and native jump impulse (`m_vecBaseVelocity[2]` + `m_hGroundEntity` + `AddOutput basevelocity`) in `source_plugin`.
- **Live Portal 1 physical verification (`portal_ml_plugin.dll` rebuilt & installed):**
  - `portal_ml_5chambers` (`records/live_physical_connections_2026-11-27.json`): exact live `IEngineTrace` geometry (`wall_rays=[1.0, 0.707, 0.5, 0.353, 0.25, ...]`, `floor_probes=[0.281, 0.281, 0.281]`), SendProp proprioception (`hp=100`, `on_ground=True`), and real physical jump arc (**`base_z=0.03u -> max_z=60.51u`, `delta_z=+60.48u`, `saw_airborne=True`, `pitch=-6.0°`**), plus **7/7 PASS** live Phase-2B regression.
  - Unmodified Valve campaign map **`testchmb_a_08`** (`records/live_testchmb_a08_physical_2026-11-27.json`): auto-discovered 9 entities & 143 portal surfaces, traced the circular glass entry elevator (`wall_rays=[0.215..0.285]`, `floor_probes=[0.37, 0.37, 0.37]`, closed door `step=-1.0`), and executed live jump (`base_z=146.56u -> max_z=207.05u`, `delta_z=+60.48u`).
- **Neuroevolved `NeuralParkourPolicy` (`champions/parkour_mlp.json`, `12->12->4`, 208 weights):**
  - Evolved from random weights (`scripts/train_parkour_skill.py --seed 11`, **100.0% validation** across 6 randomized synthetic 3D physics families: open field, tall wall slalom, low hurdle vault, toxic acid trench jump, platform gap jump, flush portal-wall entry).
  - **Controlled 13-chamber ablation (`scripts/ablate_parkour.py`, `records/ablation_parkour_2026-11-27.json`) underneath frozen `champions/seed7_decoupled.bt`:**
    - Phase 1 scripted `MoveTo`: `7/7` standard, `3/3` multi-stage (L4), **`0/3` parkour (P1–P3)**
    - Phase 2B blind `NeuralMoveTo (83w+44w)`: `7/7` standard, `3/3` multi-stage (L4), **`0/3` parkour (P1–P3)**
    - Milestone 7 sensor `NeuralParkour (208w+44w)`: **`7/7` standard, `3/3` multi-stage (L4), `3/3` parkour** (`P1` Partition Slalom 19 steps / 100 HP; `P2` Hurdle & Acid Trench 22 steps / 75 HP; `P3` Gap Jump & Timed Vault 28 steps / 100 HP) — **13/13 PASS**.
  - Honest live transfer observation (`records/live_7ch_parkour_2026-11-27.jsonl`): zero-shot `NeuralParkour` cleared **4/7 live in Portal 1** (`Ch1` 8 steps, `Ch2` 11 steps, `Ch3` 42 steps, `Ch4` 122 steps) while jumping over cubes/portals on Ch5–7 because `SkipPlayerFilter` traces knee-height loose cubes as `step_clearance=+1.0` — motivating target-aware ray filtering & Layer-4 `skill_mode` routing in Milestone 8.

---

## Upcoming Milestones (Worked One at a Time)

### Milestone 8 — Multi-Skill Routing (`skill_mode`), Native `+use` Carry/Drop & `LFM2.5-1.2B` LoRA Distillation
- Add target-aware ray filtering (ignoring the targeted `cube`/button during `Grab`/`Drop`) + Layer-4 `skill_mode` routing (`NeuralMoveTo` vs `NeuralParkour` vs future `NeuralFling`) so `NeuralParkour` hits 7/7 live in Portal 1 too.
- Replace `point_teleport` cube drops with native `+use` / `CPlayerPickupController` carry-and-drop in `source_plugin/src/motor_controller.cpp` and test live 4-layer runs (`phi-4-mini-instruct @ 4k`) on multi-room campaign maps (`testchmb_a_02` Rooms 1–2, `testchmb_a_05`).
- Build JSONL trace distillation dataset (`records/*.jsonl` -> `(symbolic_digest, semantic_action)`) and LoRA-finetune `LFM2.5-1.2B` (from `16/20` -> `20/20` matching `Phi-4-mini` at 1.25 GB).
