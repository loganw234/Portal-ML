# Changelog

All notable milestones and releases for **Portal-ML** are documented here.

## [1.0.0] — 2026-11-27

First full 4-layer core release: decoupled symbolic reasoning, neuroevolved multi-skill motor policies, a 20/20 LoRA-tuned 1.2B strategic planner, live *Portal 1* Source Engine plugin with `IEngineTrace` LiDAR, SVG path visualizer, and native `1280×720` 60fps first-person POV video capture.

### Highlights
- **Layer 1 — Perception & Generic BSP Discovery (`portal_ml/schema/`, `source_plugin/`):**
  - Ground-truth symbolic Blackboard (`BlackboardSnapshot`) + live proprioception (`health`, `on_ground`, `ducking`, `water_level`) + 13-channel egocentric `LocalSensors` (`8` wall rays, `3` floor-drop probes, `step_clearance`, `hazard_ahead`) via analytic raycasts in Python and native Source **`IEngineTrace` (`EngineTraceServer003`)** in 32-bit C++.
  - Generic BSP v20 scanner (`BspScanner`) auto-discovering gameplay entities, `trigger_hurt` hazards, multi-story `room_spawns`, `weapon_portalgun` capabilities, and portalable concrete surfaces on arbitrary Valve maps.
- **Layer 2 — Evolved Reactive Behavior Tree DSL (`portal_ml/bt/`, `portal_ml/evolution/`):**
  - Frozen 24-node champion (`champions/seed7_decoupled.bt`) evolved strictly on synthetic mock Chambers 1–4; generalizes zero-shot to held-out Chambers 5–7 (**4/5 seeds 80%** vs **1/5 20%** unconstrained; **7/10 seeds** across 10-seed causal-invariance sweep) and transfers zero-shot to live *Portal 1*.
- **Layer 3 — 6-DOF Motor Control & Multi-Skill Neuroevolved MLPs (`portal_ml/motor/`):**
  - Full 6-DOF body + interaction `MoveVector` (`forward`, `strafe`, `yaw_delta`, `pitch_delta`, `jump`, `crouch`, `use`, `fire_blue`, `fire_orange`) with native Source SendProp velocity & jump actuation (`delta_z = +60.48u` live).
  - Automatic sensor-driven `HybridMotorController` routing between `champions/moveto_mlp.json` (`83w`), `champions/parkour_mlp.json` (`208w`), and `champions/aim_mlp.json` (`44w`): **13/13 mock sweep** (`7/7` standard + `3/3` multi-stage L4 + `3/3` parkour/hazard P1–P3 vs `0/3` parkour for scripted and blind neural).
- **Layer 4 — Event-Driven Strategic Memory & 20/20 Fine-Tuned `LFM2.5-1.2B` (`portal_ml/planner/`):**
  - Rolling deduplicated `EpisodicMemory(max_entries=8)` + affordance-gated semantic subgoal `json_schema` (`place_cube_on_button`, `press_pedestal`, `go_to_exit`).
  - Controlled 20-episode ablation (`Ch 1–7` + `M1–M3`, scripted & Phase 2B neural):
    - 3-layer (`none`): `14/20` (`7/7` std, `0/3` multi-stage).
    - 4-layer `Phi-4-mini-instruct @ 4k` (3.8B, 2.49 GB): **`20/20`** (`466–481 ms/call`, `22` calls).
    - 4-layer `LFM2.5-1.2B` zero-shot (1.2B, 1.25 GB): `16/20` (`7/7` std, `1/3` M2 only).
    - 4-layer **`lfm25-1.2b-layer4` LoRA `Q8_0` GGUF** (1.2B, `1.16 GiB`, `scripts/build_layer4_dataset.py` + `scripts/train_lfm25_lora.py` + `scripts/export_lfm25_gguf.py`): **`20/20` PASS** (`62/66/78` scripted @ `285.8 ms/call`, `81/82/113` neural @ `259.7 ms/call`, `22` theoretical-min calls, `0` fallbacks, beating both `Rule` and `Phi-4-mini` on M3 by 16–36 steps).
- **Live *Portal 1* Verification & Tooling:**
  - **7/7 PASS** live on `portal_ml_5chambers` with all 4 layers + Multi-Skill Auto MLPs powered by `lfm25-1.2b-layer4` (`13` calls / `4.15s` = `319 ms/call`) and `Phi-4-mini @ 4k` (`6.43s`).
  - **PASS** live on unmodified Valve **`testchmb_a_02` Room 1 (Test Chamber 04)** (3/3 repeatable in `149` steps with post-trigger pad pinning; `850 ms` total LLM time on `lfm25-1.2b-layer4`).
  - Zero-dependency SVG path visualizer (`scripts/render_paths.py`), native `1280×720` 60fps POV video capture (`--record-video`), one-click Windows installer (`scripts/install_plugin.ps1`), and 41-test verification gate (`python tools/check.py --control`).

### Known `v1.0.0` Scope Boundaries (Queued for `v1.1+`)
- Multi-bend campaign corridors (`testchmb_a_02` Room 2) need an `IEngineTrace`-validated 2D/3D `A*` waypoint pathfinder (Milestone 11).
- Same-elevation fizzler barriers (`testchmb_a_05`) need horizontal fizzler zone splitting (Milestone 11).
- Energy pellets, turrets, momentum flinging, native `+use` physics carry/throw, and optional pixel vision are future milestones.
