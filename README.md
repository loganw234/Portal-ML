# Portal-ML

A hybrid neuroevolutionary reasoning agent for *Portal* (Source Engine) that decouples **causal reasoning** (evolved as a readable Behavior Tree DSL over a symbolic Blackboard) from **motor execution** (deterministic Phase-1 primitives now, swappable for neuroevolved neural networks in Phase 2).

The core claim is architectural: separating *what* subgoal entity to pursue from *how* to reach and interact with `target` lets an agent trained strictly on Chambers 1–4 generalize zero-shot to a held-out rotated puzzle (Chamber 5) that combines cubes, buttons, and portals for the first time.

## Status (2026-11-26)

- **3-layer PoC and 5-chamber mock 3D simulator built and measured.** Pure Python 3.12 (zero external dependencies) runs at **1,978 episodes/sec** (**42,734 ticks/sec**) on one desktop core (`python scripts/benchmark_sim.py --episodes 5000`).
- **Decoupled vs unconstrained ablation measured (5 seeds, pop 120, max 120 gens, trained strictly on Chambers 1–4):**
  - `decoupled` (`SEQUENCE(goal_selector, execution)`): **4/5 seeds** solved all 4 training chambers; **2/5 seeds** passed held-out Chamber 5 zero-shot in **44–45 steps** (mean size 21.8 AST nodes).
  - `unconstrained` (flat single BT): **0/5 seeds** solved Chambers 1–4 (all stalled at 2/4); **0/5** passed Chamber 5.
- **Live Portal 1 (`appID=400`) map, C++ server plugin, and sim-to-Source transfer verified (Milestone 1, 2026-11-26).**
  - Single-map 5-chamber `.vmf` compiled cleanly through Portal 1 `vbsp`, `vvis`, and `vrad` with zero leaks (512 triangles, 5 airtight chambers separated by 2048u void gaps, flush 1u concrete portal panels, and `point_teleport` reset targets).
  - 32-bit x86 MSVC C++17 server plugin (`ISERVERPLUGINCALLBACKS003`, zero `tier0.lib`/`tier1.lib` link dependencies) compiled cleanly and verified live over `127.0.0.1:27500`.
  - **Mock-evolved champion (`champions/seed7_decoupled.bt`, 24 nodes, trained strictly on Chambers 1–4 in Python) transferred zero-shot to live Portal 1 and cleared 5/5 chambers** in ~11.0s total (`records/live_seed7_2026-11-26.jsonl`):
    - Chamber 1: PASS (11 steps, 1.16s, reset 104.5ms)
    - Chamber 2: PASS (27 steps, 2.84s, reset 96.5ms)
    - Chamber 3: PASS (20 steps, 2.10s, reset 105.3ms; cube dropped 7u from button center)
    - Chamber 4: PASS (11 steps, 1.08s, reset 103.5ms; live `prop_portal` traversal across pit)
    - Chamber 5 (held-out): **PASS zero-shot** (38 steps, 3.86s, reset 104.2ms; bidirectional portal crossing with cube, dropped 1.2u from button center)
- **Phase 2 neuroevolved neural motor primitives verified on mock AND live Portal 1 (Milestone 2, 2026-11-26).**
  - Evolved two small egocentric MLPs from random Gaussian weights (`python scripts/train_phase2_motor.py --seed 7`): `champions/moveto_mlp.json` (`6->8->3`, **83 weights**, 100% validation in 25.9s) and `champions/aim_mlp.json` (`4->6->2`, **44 weights**, 100% validation in 7.7s).
  - Plugged underneath the **frozen** `champions/seed7_decoupled.bt` (zero retraining of Layer 2):
    - **Phase 2A (neural `MoveTo`, 83w):** **5/5 mock** (16/18/27/10/47 steps) and **5/5 live Portal 1** (11/12/23/10/51 steps; `records/live_phase2a_2026-11-26.jsonl`).
    - **Phase 2B (neural `MoveTo` + neural `ShootPortal Aim`, 127w total):** **5/5 mock** (16/18/27/20/67 steps) and **5/5 live Portal 1** (12/12/24/24/85 steps; `records/live_phase2b_2026-11-26.jsonl`).

- **Generalization hardening, expanded 3-chamber held-out suite, and 7/7 live Portal 1 verification (Milestone 3, 2026-11-26).**
  - Expanded the held-out suite to three unseen chambers (`HELD_OUT_CHAMBERS = (5, 6, 7)`): Chamber 5 (90° rotated cube across gap), Chamber 6 (cube at spawn, floor button across gap with opposite-facing walls), and Chamber 7 (3-zone, 2-pit portal relay requiring two mid-episode portal re-links).
  - Diagnosed zero-shot failures to two root causes: an asymmetric unguarded `Drop()` primitive in `BehaviorTreeEvaluator` and two training-set spurious correlations (`wall_exists <=> target_unreachable` in Chamber 3; `portals_active <=> portals_bridge(target)` in Chamber 4).
  - Fixed `Drop()` target-locality symmetry + added two training-only causal invariance augmentations strictly on Training Chambers 3 & 4 (zero held-out leakage). Across 10 seeds (`[1, 2, 3, 5, 7, 11, 13, 17, 21, 42]`, pop 120, gens 120):
    - Baseline decoupled: `train_4/4 = 8/10`, `Ch5 = 3/10`, `Ch6 = 3/10`, `Ch7 = 3/10`, `all 3 held-out = 3/10`.
    - `+ Drop()` symmetry: `train_4/4 = 10/10`, `Ch5 = 6/10`, `Ch6 = 6/10`, `Ch7 = 5/10`, `all 3 = 5/10`.
    - `+ Causal invariance augmentation`: `train_4/4 = 9/10`, `Ch5 = 7/10`, `Ch6 = 8/10`, `Ch7 = 7/10`, **`all 3 held-out = 7/10` (2.33x baseline)**.
  - Canonical 5-seed ablation (`[1, 7, 13, 21, 42]`): `decoupled` solved **5/5 train** and **4/5 (80%) across all three held-out chambers (5, 6, 7)** vs `unconstrained` **1/5 (20%)**.
  - Synced Chambers 6 & 7 into `portal_ml_5chambers.bsp` (956 triangles, zero leaks) and `blackboard_extractor.cpp`, and verified **7/7 chambers PASS live in Portal 1** for BOTH Phase 1 scripted (`11/27/19/11/39/39/47` steps, 1.50s total; `records/live_7ch_phase1_2026-11-26.jsonl`) and Phase 2B neural `MoveTo(83w) + Aim(44w)` (`11/10/20/22/61/72/77` steps, 2.23s total, ~7.2ms resets; `records/live_7ch_phase2b_2026-11-26.jsonl`).
- **Real Portal 1 campaign feasibility audited (`docs/REAL_GAME_ANALYSIS.md`, 2026-11-26).** Audited all 24 real Valve `.bsp` maps directly from `portal_pak_dir.vpk` via `scripts/audit_campaign_maps.py`, identifying Tier-1 real campaign targets (`testchmb_a_00`, `a_01`, `a_02`, `a_05`, `a_06`) and the four upgrades required to scale toward full-game completion.
- **Native SendProp physics, live C++ milestones, and generic live-BSP discovery verified on unmodified `testchmb_a_02` (Milestone 4, 2026-11-26).**
  - Replaced per-tick `ent_fire !player AddOutput basevelocity` console strings with direct SendProp `m_vecBaseVelocity` + Portal 1 `FL_BASEVELOCITY` (`1<<23` under `-DPORTAL=1`) actuation and live C++ milestone reward reporting (`cube_grabbed`, `button_pressed`, `portal_*`, `portal_traversed`, `exit_reached`).
  - Verified **7/7 live Portal 1 PASS** for BOTH Phase 1 scripted (`13/35/22/11/51/40/43` steps, 2.66s total; `records/live_m4_native_phase1_2026-11-26.jsonl`) and Phase 2B neural `MoveTo(83w) + Aim(44w)` (`12/58/21/21/69/62/98` steps, 2.79s total, ~122 steps/sec, ~8.1ms resets; `records/live_m4_native_phase2b_2026-11-26.jsonl`).
  - Built C++ `BspScanner` (`source_plugin/src/bsp_scanner.cpp`) + Python mirror (`portal_ml/vmf/bsp_reader.py`) triggered automatically on `LevelInit` for arbitrary maps. Tested live on Valve's unmodified campaign map **`testchmb_a_02`** (`records/live_testchmb_a02_2026-11-26.json`): auto-discovered **9 gameplay entities** (cubes, composite `button_top_reference` floor buttons, exit doors) and **293 portalable concrete surfaces** (walls, floors, ceilings), streaming live `BlackboardSnapshot`s and native `MoveVector` motion over TCP IPC.
  - Documented the proposed **4th-layer small-LLM strategic memory & subgoal planner** in [docs/FOUR_LAYER_ARCHITECTURE.md](docs/FOUR_LAYER_ARCHITECTURE.md).

What comes next is tracked in [docs/ROADMAP.md](docs/ROADMAP.md), [docs/REAL_GAME_ANALYSIS.md](docs/REAL_GAME_ANALYSIS.md), and [docs/FOUR_LAYER_ARCHITECTURE.md](docs/FOUR_LAYER_ARCHITECTURE.md): live real-campaign benchmarking on `testchmb_a_02` (Milestone 5) and Layer-4 small-LLM strategic planning (Milestone 6).

## The front door

    python tools/check.py
    python tools/check.py --control

Runs every gate (personal-path guard, Python↔C++ chamber 1–7 coordinate parity, held-out Chambers 5–7 isolation, `.vmf` integrity, Phase-2 checkpoint schema validation, and the 20-test unit/integration suite) and ends with `VERDICT: PASS`. Use `--quick` to skip the full evolution test during fast edits.

## Architecture

1. **Layer 1 — Perception -> Symbolic Blackboard** (`portal_ml/schema/blackboard.py`): Ground-truth JSON snapshot every tick in Source Hammer units (player position/velocity/yaw/zone/held object, entities, door/button states, portalable surface normals, active portal transforms).
2. **Layer 2 — Evolved Reasoning Layer** (`portal_ml/bt/`, `portal_ml/evolution/`): Readable Behavior Tree DSL (`SELECTOR`, `SEQUENCE`, `CONDITION`, `ACTION`) evolved by structural & semantic AST mutations and subtree crossover.
3. **Layer 3 — Motor Execution Layer** (`portal_ml/motor/primitives.py`): Deterministic Phase-1 primitives (`MoveTo`, `Grab`, `Drop`, `Press`, `ShootPortal`) plus the `MotorPolicy` protocol for Phase-2 neural controllers.

## 7-Chamber Testbed (4 Training + 3 Held-Out)

All 7 chambers share one `.vmf` coordinate space separated by 2048-unit void gaps along Y:

| Chamber | Name | Role | Validated Capability |
|---|---|---|---|
| 1 | The Hallway | Training | Bridge, fixed-timestep stepping, navigation |
| 2 | The Pedestal | Training | 2-step causality (timed pedestal -> open exit) |
| 3 | Box on Button | Training | Object permanence & state binding (grab -> carry -> drop -> exit) |
| 4 | The Simple Gap | Training | Non-Euclidean portal placement & traversal |
| 5 | Held-Out Generalization Test | **Held-Out** | Rotated 90° combo (cube across pit, floor button & closed exit at spawn) |
| 6 | Held-Out: Button Across Gap | **Held-Out** | Opposite asymmetry (cube & exit at spawn, floor button across pit, opposite-facing walls) |
| 7 | Held-Out: Three-Zone Relay | **Held-Out** | 3 disconnected islands separated by 2 pits; requires two mid-episode portal re-links (`0->2->1->0`) |

## Running

```powershell
# Run evolution on the fast mock simulator and test Chamber 5 zero-shot
python scripts/train_mock.py --seed 7 --gens 80

# Reproduce the 5-seed decoupled vs unconstrained ablation
python scripts/ablation_study.py --seeds 1 7 13 21 42 --pop 120 --gens 120

# Benchmark mock simulator throughput
python scripts/benchmark_sim.py --episodes 5000

# Regenerate the 5-chamber .vmf and compile scripts
python scripts/generate_vmf.py
maps\compile_map.bat
```

## Live Portal 1

See [source_plugin/README.md](source_plugin/README.md) for building `portal_ml_plugin.dll` (32-bit MSVC against Valve's `source-sdk-2013` public headers). Once installed into `<Portal>\portal\addons\`:

```powershell
.\scripts\launch_portal.ps1
python scripts/test_live_portal.py
python scripts/train_mock.py --backend source --host 127.0.0.1 --port 27500
```

## Layout

| Folder | Contents |
|---|---|
| `portal_ml/` | Python library: `schema/` (Blackboard), `bt/` (DSL & evaluator), `motor/` (primitives), `sim/` (3D mock env & chamber specs), `evolution/` (GP trainer), `bridge/` (TCP IPC client), `vmf/` (`.vmf` builder) |
| `source_plugin/` | 32-bit C++17 Source Engine server plugin (`ISERVERPLUGINCALLBACKS003`) |
| `maps/` | Generated `portal_ml_5chambers.vmf` and `vbsp`/`vvis`/`vrad` compile scripts |
| `scripts/` | CLI entry points: training, ablation, benchmarking, VMF generation, Portal launcher, live smoke test |
| `tools/` | Front-door verification gate (`check.py`) |
| `tests/` | Unit and integration tests (including loopback TCP IPC bridge and evolution generalization) |
| `docs/` | Project roadmap ([docs/ROADMAP.md](docs/ROADMAP.md)) |

## Not affiliated with Valve

Portal-ML is an independent research project. It is not affiliated with, or endorsed by, Valve Corporation. *Portal* and *Source* are trademarks of Valve Corporation.

## Licence

MIT, [LICENSE](LICENSE).
