# Portal-ML

A hybrid neuroevolutionary reasoning agent for *Portal* (Source Engine) that decouples **causal reasoning** (evolved as a readable Behavior Tree DSL over a symbolic Blackboard) from **motor execution** (deterministic Phase-1 primitives now, swappable for neuroevolved neural networks in Phase 2).

The core claim is architectural: separating *what* subgoal entity to pursue from *how* to reach and interact with `target` lets an agent trained strictly on Chambers 1–4 generalize zero-shot to a held-out rotated puzzle (Chamber 5) that combines cubes, buttons, and portals for the first time.

## Status (2026-11-26)

- **3-layer PoC and 5-chamber mock 3D simulator built and measured.** Pure Python 3.12 (zero external dependencies) runs at **1,978 episodes/sec** (**42,734 ticks/sec**) on one desktop core (`python scripts/benchmark_sim.py --episodes 5000`).
- **Decoupled vs unconstrained ablation measured (5 seeds, pop 120, max 120 gens, trained strictly on Chambers 1–4):**
  - `decoupled` (`SEQUENCE(goal_selector, execution)`): **4/5 seeds** solved all 4 training chambers; **2/5 seeds** passed held-out Chamber 5 zero-shot in **44–45 steps** (mean size 21.8 AST nodes).
  - `unconstrained` (flat single BT): **0/5 seeds** solved Chambers 1–4 (all stalled at 2/4); **0/5** passed Chamber 5.
- **Live Portal 1 (`appID=400`) map and C++ server plugin built and verified.**
  - Single-map 5-chamber `.vmf` compiled cleanly through Portal 1 `vbsp`, `vvis`, and `vrad` with zero leaks (512 triangles, 5 airtight chambers separated by 2048u void gaps for sub-millisecond in-place resets).
  - 32-bit x86 MSVC C++17 server plugin (`ISERVERPLUGINCALLBACKS003`, zero `tier0.lib`/`tier1.lib` link dependencies) compiled cleanly and verified against live Portal 1 over `127.0.0.1:27500`.

What comes next is tracked in [docs/ROADMAP.md](docs/ROADMAP.md): sim-to-Source live replay & physics calibration, Phase-2 neuroevolved neural motor primitives, zero-shot generalization hardening, and native `CUserCmd` control.

## The front door

    python tools/check.py
    python tools/check.py --control

Runs every gate (personal-path guard, Python↔C++ chamber coordinate parity, Chamber 5 held-out isolation, `.vmf` integrity, and the 14-test unit/integration suite) and ends with `VERDICT: PASS`. Use `--quick` to skip the full evolution test during fast edits.

## Architecture

1. **Layer 1 — Perception -> Symbolic Blackboard** (`portal_ml/schema/blackboard.py`): Ground-truth JSON snapshot every tick in Source Hammer units (player position/velocity/yaw/zone/held object, entities, door/button states, portalable surface normals, active portal transforms).
2. **Layer 2 — Evolved Reasoning Layer** (`portal_ml/bt/`, `portal_ml/evolution/`): Readable Behavior Tree DSL (`SELECTOR`, `SEQUENCE`, `CONDITION`, `ACTION`) evolved by structural & semantic AST mutations and subtree crossover.
3. **Layer 3 — Motor Execution Layer** (`portal_ml/motor/primitives.py`): Deterministic Phase-1 primitives (`MoveTo`, `Grab`, `Drop`, `Press`, `ShootPortal`) plus the `MotorPolicy` protocol for Phase-2 neural controllers.

## 5-Chamber Micro-Curriculum

All 5 chambers share one `.vmf` coordinate space separated by 2048-unit void gaps along Y:

| Chamber | Name | Role | Validated Capability |
|---|---|---|---|
| 1 | The Hallway | Training | Bridge, fixed-timestep stepping, navigation |
| 2 | The Pedestal | Training | 2-step causality (timed pedestal -> open exit) |
| 3 | Box on Button | Training | Object permanence & state binding (grab -> carry -> drop -> exit) |
| 4 | The Simple Gap | Training | Non-Euclidean portal placement & traversal |
| 5 | Held-Out Generalization Test | **Held-Out** | Rotated 90° combo (cube across pit, floor button & closed exit at spawn); strict pass/fail + step count |

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
