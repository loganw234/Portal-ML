# Working on Portal-ML

Short, and only what is non-obvious. [README.md](README.md) says what the project is; this says what will bite.

## Start here: one command

    python tools/check.py

Read the last line, not just the exit code: `VERDICT: PASS (<N> gates)`. Run `python tools/check.py --control` to verify every gate against a planted fault (`CONTROL-DETECTED`). Zero third-party Python dependencies; Python 3.11+ runs everything directly.

## Rules that hold the claims

- **Chamber 5 is strictly held out.** `TRAIN_CHAMBERS` in `portal_ml/evolution/fitness.py` is `(1, 2, 3, 4)` and `HELD_OUT_CHAMBER` is `5`. Chamber 5 must never contribute to selection, shaping, or early-stopping; it is evaluated once on the final champion strictly on pass/fail and step count.
- **No hand-coded solver in evolution.** `portal_ml/bt/reference_trees.py` is an oracle for simulator, VMF, and bridge sanity tests only. `EvolutionTrainer` starts from random ASTs.
- **Single source of truth for chamber geometry.** All 5 chambers live in `portal_ml/sim/chambers.py` in Source Hammer units (separated by 2048u void gaps along Y). Both `portal_ml/vmf/generator.py` and `source_plugin/src/blackboard_extractor.cpp` must match those coordinates 1:1; `tools/check.py` verifies the C++ layout table against `chambers.py`.
- **Same `PortalEnv` interface across backends.** `MockPortalEnv` and `SourcePortalEnv` share `reset(chamber_id) -> BlackboardSnapshot` and `step(command) -> StepResult` over identical JSON snapshots and Layer-3 commands.
- **Keep `INTERACT_RADIUS <= BUTTON_TRIGGER_RADIUS`.** In `predicates.py` (`44.0`) and `mock_env.py` (`52.0`), dropping a cube once `in_range(floor_button)` holds must land inside the button trigger volume.
- **No personal paths in tracked files.** `tools/check.py` fails on user-specific paths (e.g. `Users\<name>`).
- **Commit and push only on Logan's word, each time.** Include the `Co-Authored-By:` trailer naming the assisting model on every commit.

## Traps

1. **Portal 1 is 32-bit x86 Win32 (`hl2.exe` PE machine `0x014C`).** Build `source_plugin` with MSVC `-A Win32` (`Visual Studio 16 2019` or compatible MSVC x86 toolset). A 64-bit DLL or MinGW ABI will fail to load in `engine.dll`.
2. **Portal 1 vs Portal 2 entity names.** Portal 1 (`appID=400`, Source 2013 SP) does *not* have `prop_floor_button`, `prop_button`, or `prop_testchamber_door`. Use `func_weight_button`, `func_button`, `func_door`, `prop_physics` (`models/props/metal_box.mdl`), `prop_portal`, and `weapon_portalgun`.
3. **Water shaders do not seal the void in `vbsp.exe`.** Pit floors must be backed by a solid opaque brush (`METAL/METAL_MODULAR_FLOOR001`), never a water/slime shader alone, or `vbsp` reports a leak and `vvis` fails.
4. **Zero `tier0.lib` / `tier1.lib` dependency in `portal_ml_plugin.dll`.** Export `CreateInterface` directly (`ISERVERPLUGINCALLBACKS003`) and avoid `NO_MALLOC_OVERRIDE=1` in MSVC builds (`memdbgon.h` falls into a POSIX `memalign` branch when it is set).
5. **Line endings.** Git on Windows often enables `core.autocrlf=true`; `.gitattributes` pins LF everywhere except `.bat` files.
