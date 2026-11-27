# Real Portal 1 Campaign Analysis & Full-Game Feasibility

*Audited 2026-11-26 against the live Steam Portal 1 installation (`portal/portal_pak_dir.vpk`, 24 campaign `.bsp` maps) via [audit_campaign_maps.py](../scripts/audit_campaign_maps.py).*

## 1. Executive Answer

**Yes — both goals are viable:**

1. **Using real Portal 1 campaign chambers as advanced test suites:** Early campaign chambers (`testchmb_a_00`, `testchmb_a_01`, `testchmb_a_02`, `testchmb_a_05`, `testchmb_a_06`) rely on the same core puzzle mechanics our 3-layer architecture already models (cubes, weighted floor buttons, portals, doors, plus emancipation fizzlers) and can be targeted directly once `blackboard_extractor.cpp` replaces its static chamber layout table with generic live-BSP entity and portalable-surface discovery.
2. **Using this 3-layer decoupled architecture as a base for an agent capable of beating the full game over time:** The separation between **symbolic state -> evolved/learned causal Behavior Tree -> neuroevolved motor skills** is a strong foundation for full-game play—significantly more sample-efficient and interpretable than end-to-end pixel RL for multi-step causal puzzles. However, beating the entire 19-chamber + escape campaign requires **four concrete architectural upgrades** beyond the current PoC, detailed in Section 4.

---

## 2. Ground-Truth Entity Audit of All 24 Portal 1 Campaign Maps

Running `python scripts/audit_campaign_maps.py` extracts the raw BSP entity lump (lump 0) directly from all 24 maps inside `portal_pak_dir.vpk` (16 main test chambers, 6 `_advanced` variants, and 3 escape/boss maps).

### Critical Engine-Level Finding
Real Valve campaign maps **do not** use `func_weight_button` (the simplified single-brush entity used in our custom `portal_ml_5chambers.vmf`). Instead, Portal 1's shipped maps build:
- **Weighted floor buttons** from composite prefabs: `trigger_multiple` + `filter_activator_name` + `func_door` (moving button pad) + `prop_dynamic` base.
- **Chamber exits** from `func_door` airlocks leading into a `func_tracktrain` elevator + `trigger_changelevel`.
- **Fizzlers (Emancipation Grills)** from `trigger_portal_cleanser` (which vaporize held cubes and clear active portals when crossed).

### Campaign Difficulty Tiers

| Tier | Maps | Mechanics Present (from BSP entity lumps) | Gap from Current PoC |
|---|---|---|---|
| **Tier 1: Core Cube / Button / Portal** | `testchmb_a_00`, `a_01`, `a_02`, `a_05`, `a_06` | `prop_physics` (2–12), pre-placed `prop_portal` (1–8), `weapon_portalgun`, `trigger_portal_cleanser` (1–2), `func_door` (4–13), `func_tracktrain` (2–3), `trigger_multiple` (0–9) | Generic entity/button/exit discovery + BSP portalable-surface extraction + fizzler awareness + intro fling (`a_06`) |
| **Tier 2: Energy Pellets & Moving Platforms** | `testchmb_a_03`, `a_04`, `a_07`, `a_08` (+adv), `a_09` (+adv), `a_10` (+adv), `a_13` (+adv) | Adds `point_energy_ball_launcher` (1–3) + `func_tracktrain` moving lifts/platforms (3–8) + multi-stage fizzlers (`a_10` has 5–6 cleansers) | Pellet trajectory/catcher state in Blackboard + timing/intercept motor skill + dynamic moving-zone tracking |
| **Tier 3: Turrets, Escape & GLaDOS Boss** | `testchmb_a_11` (+adv), `a_14` (+adv), `a_15`, `escape_00`, `escape_01`, `escape_02` | Adds `npc_portal_turret_floor` (4–12), `npc_rocket_turret` (1), `prop_glados_core` (4), 26–79 physics props, multi-room verticality | Threat/line-of-sight perception, combat/knockover & rocket-redirect skills, long-horizon subgoal planning |

### Recommended First Real-Game Benchmark Targets

1. **`testchmb_a_02` (Chamber 04/05 — Cube, Button, Portal Gun, Fizzler):** Only 2 `prop_physics`, 4 `prop_portal`, 2 `trigger_portal_cleanser`, 7 `func_door`, 9 `trigger_multiple`. Closest direct real-game analogue to our Chambers 3–5.
2. **`testchmb_a_00` (Relaxation Vault + Chambers 00–01):** Tests scripted intro portal timing, cube-on-button placement (`12` props, `8` pre-placed portals), and elevator entry without requiring the portal gun.
3. **`testchmb_a_05` (Chambers 08/09 — Dual-Button / Cube / Fizzler):** Clean bridge to multi-button / multi-cube reasoning (`5` props, `2` cleansers, `4` doors).
4. **`testchmb_a_06` (Chamber 10 — First Momentum Fling):** First real test of Phase-2 neuroevolved continuous physics control (vertical drop into floor portal -> wall launch).

---

## 3. Why the 3-Layer Decoupled Architecture Scales Better Than Pure End-to-End RL

- **Causal credit assignment over long horizons:** Real Portal chambers span hundreds to thousands of ticks with sparse rewards (open door -> enter elevator -> ride lift -> next room). Evolving or searching over a symbolic Behavior Tree / hierarchical planner compresses 1,000+ low-level physics ticks into 5–15 causal decisions.
- **Reusable motor skills across all 19 chambers:** Once a small neuroevolved network learns `MoveTo`, `AimPortal`, `Fling`, or `CatchPellet` (each 40–300 weights), the exact same skill executes unchanged on every map regardless of wall textures, lighting, or room orientation—as already demonstrated by our frozen 24-node BT + 83w/44w neural motor passing Chambers 1–7 zero-shot.
- **Inspectable failure diagnosis:** Every failure on a new chamber isolates cleanly into one of three buckets: (a) Perception/Blackboard missed an entity or surface, (b) Reasoning/BT lacked a causal branch or subgoal ordering, or (c) Motor primitive failed a physics maneuver.

---

## 4. Four Required Upgrades to Beat the Full Game

To grow `Portal-ML` from the 7-chamber PoC into an agent that solves real Portal 1 campaign chapters:

### Upgrade 1 — Generic Live-Map Blackboard Extractor (Replaces Static `kLayouts`)
- **Entity discovery:** Scan all live server edicts dynamically by class and model (`models/props/metal_box.mdl`, `models/props/button_base_reference.mdl`, `trigger_portal_cleanser`, `point_energy_ball_launcher`, `npc_portal_turret_floor`, `trigger_changelevel`, `func_tracktrain`). Detect composite floor buttons by pairing `trigger_multiple` + `func_door` button pads.
- **BSP portalable-surface discovery:** Parse the loaded map's `.bsp` faces/texinfo (or server-side `IEngineTrace` sweeps) at `LevelInit` to build a live catalog of portalable concrete surfaces (`CONCRETE/CONCRETE_MODULAR_*`, `surfaceprop == concrete`, excluding `NOPORTAL` flags), including **floors, ceilings, and angled panels**—not just vertical walls.
- **Dynamic connectivity graph:** Replace static `WalkableZone` AABBs with an automatic navmesh/flood-fill connectivity graph computed from BSP floor geometry + moving `func_tracktrain` platform links.

### Upgrade 2 — Parameterized Multi-Entity Symbolic Reasoning
- Current `WorkingMemory` holds a single scalar `(target_role, target_entity_id)` and resolves `first_of_kind(...)`.
- Full campaign chambers require **indexed / relational binding** (e.g., `nearest_unpressed(floor_button)`, `unplaced(cube)`, `active_pellet_catcher`, `blocking_fizzler`, `next_elevator_door`) plus a small **subgoal stack** (or hierarchical BT library) so multi-room maps (`testchmb_a_15`, `escape_00..02`) don't blow up a single flat AST.

### Upgrade 3 — Expanded Library of Phase-2 Neuroevolved Motor Primitives
Keep each skill small, specialized, and independently trained in isolation (just like our 83w `MoveTo` and 44w `Aim` MLPs):
1. `Navigate3D` (3D pathing around obstacles, ramps, stairs, moving platforms, and avoiding fizzlers while holding cubes).
2. `AimAndShootPortal3D` (full pitch+yaw aiming onto arbitrary walls, floors, ceilings, and angled fling panels, including line-of-sight verification).
3. `MomentumFling` (continuous air-strafe + timing control for vertical portal drops and double-flings in `testchmb_a_06`, `a_10`, `a_15`).
4. `InterceptPellet` (timing portal placement to redirect `prop_energy_ball` into receivers).
5. `NeutralizeTurret` (portal-under-turret or flank-and-tip maneuvers for `testchmb_a_11`, `a_14`, `escape_01`, `escape_02`).

### Upgrade 4 — Real-Engine Physics Integration (`IPhysicsObject` + Native `CUserCmd`)
- Replace the Milestone 1/2 `ent_teleport` + `basevelocity` shortcuts with native Source server interfaces: `CBasePlayer::PickupObject` / `DropObject`, `IPhysicsObject::SetPosition`, and direct `CUserCmd` input injection (`forwardmove`, `sidemove`, `buttons` including `IN_JUMP`, `IN_USE`, `IN_ATTACK`, `IN_ATTACK2`) so jumping, crouching, air-strafing, fizzler interaction, and momentum conservation through portals are 100% authentic Source physics.

---

## 5. Concrete Incremental Path (Next Milestones)

| Phase | Target | Concrete Deliverable |
|---|---|---|
| **Milestone 4** | Native physics + Generic Extractor | Replace `ent_teleport` / `basevelocity` with native `CUserCmd` + `IPhysicsObject`; add BSP-driven entity & portalable-surface discovery (`scripts/audit_campaign_maps.py` -> C++ `LevelInit` parser). |
| **Milestone 5** | First Real Campaign Chamber (`testchmb_a_02`) | Run a zero-edit real Valve map (`map testchmb_a_02`) over the existing TCP bridge with generic Blackboard extraction and solve it live. |
| **Milestone 6** | Tier-1 Campaign Suite (`a_00`, `a_01`, `a_02`, `a_05`, `a_06`) | Add fizzler-aware planning, floor/ceiling portal surfaces, and a neuroevolved `MomentumFling` primitive. |
| **Milestone 7+** | Tier-2 & Tier-3 (`a_03..a_15`, `escape_00..02`) | Add pellet/platform/turret Blackboard schemas, relational BT subgoal stack, and specialized Phase-2 skills. |
