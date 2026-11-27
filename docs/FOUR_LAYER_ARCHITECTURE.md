# Four-Layer Architecture: Adding a Small-LLM Long-Term Goal & Memory Layer

*Design analysis (2026-11-26) evaluating a Layer-4 strategic planner / episodic memory module on top of Portal-ML's verified 3-layer stack.*

## 1. Why a 4th Layer Makes Sense Here

The existing 3-layer stack solves single-room and 3-zone causal puzzles cleanly at high speed (`7/7` chambers live in Portal 1 in 1.5–2.2s):

| Layer | Mechanism | Cadence | Responsibility |
|---|---|---|---|
| **Layer 1 — Perception** | Symbolic Blackboard + BSP Scanner | 66 Hz (server) / 10 Hz (step) | Ground-truth entity states, zones, portal transforms, concrete surfaces, fizzlers |
| **Layer 2 — Reactive Reasoning** | Evolved Behavior Tree DSL (`24–33` nodes) | 10 Hz (`< 0.05 ms/tick`) | Local causal execution, portal bridging/re-linking, reactive grab/drop/press |
| **Layer 3 — Motor Control** | Scripted or Neuroevolved MLPs (`44–83` weights) | 10–66 Hz (`< 0.1 ms/tick`) | Continuous body-frame steering, aiming, future flinging / pellet / turret skills |
| **Layer 4 — Strategic Memory (Proposed)** | Small Specialized LLM (`0.5B–3B`, local, GBNF-constrained) | **Event-driven (`0.1–0.5 Hz`)** | Multi-room subgoal decomposition, persistent episodic memory, stall recovery, skill-mode selection |

Where pure evolved BTs begin to hit scaling walls in full Portal 1 campaign maps (`testchmb_a_05`, `a_10`, `a_15`, `escape_00..02`):
1. **Multi-entity disambiguation:** Real maps have 2+ cubes, 2–3 buttons, multiple rooms, and fizzlers (e.g., `testchmb_a_02` has 2 cubes, 3 floor buttons, 2 exit doors, and 2 fizzlers across two stacked test rooms). A flat evolved BT grows exponentially when forced to encode arbitrary multi-stage sequences.
2. **Cross-attempt / episodic memory:** If walking through a `trigger_portal_cleanser` vaporizes a held cube, or a timed door closes before arrival, a stateless reactive BT repeats the exact same failing loop. Layer 4 remembers *"fizzler between zone A and zone B destroyed `box_1` -> retrieve cube on far side or portal-bypass fizzler"* and changes strategy.
3. **Long-horizon subgoal ordering:** Real campaign chambers require decomposing an unseen room into an ordered subgoal queue (`get_portalgun` -> `open_middle_door` -> `ride_elevator` -> `press_button_a` -> `press_button_b` -> `enter_exit_elevator`).

---

## 2. Why a Small Specialized Model (`0.5B–3B`) Is the Right Fit

Rather than calling a giant frontier model every frame, a **small local model (0.5B–3B parameters, Q4/Q8 quantized)** running locally via `llama.cpp` / ONNX / Ollama with **grammar-constrained structured decoding (GBNF / JSON schema)** is ideal:

- **Event-driven latency budget:** Layer 4 is **never** on the 10–66 Hz tick loop. It is invoked only on discrete **trigger events** (`room_entered`, `subgoal_completed`, `stall_detected`, `cube_fizzled`, `player_died`) — typically **2–8 calls per chamber**. At ~50–150 ms per 40-token structured JSON reply on a local GPU (or ~300 ms CPU), overhead is negligible.
- **Zero hallucinated actions via grammar constraints:** The output grammar restricts Layer 4 to valid entity IDs currently present in Layer 1's Blackboard and valid BT skill modes / subgoal primitives. It cannot invent controls or emit unparseable syntax.
- **Specialization via trace distillation:** Every live run already records structured JSONL traces (`records/*.jsonl`). Failed stalls, human/oracle annotations, and successful multi-room sequences can directly LoRA-finetune a 1B–3B model specifically for Portal symbolic causal reasoning.
- **Clean scientific ablation:** Because Layer 4 communicates exclusively through a typed `StrategicDirective` injected into Layer 2's `WorkingMemory`, we can run controlled ablations comparing **3-layer (BT + neural motor)** vs **4-layer (Small LLM + BT + neural motor)** on identical live campaign maps.

---

## 3. Proposed Layer-4 <-> Layer-2 Interface Contract

### Input to Layer 4 (Compact Symbolic Digest, `< 350 tokens`)
Never feed raw per-tick telemetry. Summarize Layer 1 + Layer 2 state only when a trigger fires:

```json
{
  "event": "stall_detected",
  "map": "testchmb_a_02",
  "player_zone": 0,
  "held_entity": null,
  "entities": [
    {"id": "box", "kind": "cube", "zone": 0, "active": false},
    {"id": "button_top_a_a02", "kind": "floor_button", "zone": 0, "active": false},
    {"id": "exit_door_left_a02", "kind": "exit_door", "zone": 0, "active": false}
  ],
  "hazards": [
    {"id": "a02-middle_trigger_fryer_rm2", "kind": "fizzler", "between_zones": [0, 1]}
  ],
  "active_subgoal": {"op": "place_cube_on_button", "cube_id": "box", "button_id": "button_top_a_a02"},
  "episodic_memory": [
    "t=42: box dropped at (-452,190) activated button_top_a_a02 -> exit_door_left_a02 opened"
  ]
}
```

### Output from Layer 4 (Grammar-Constrained `StrategicDirective`, `< 60 tokens`)
```json
{
  "subgoal_queue": [
    {"op": "bind_target", "role": "exit_door", "entity_id": "exit_door_left_a02"},
    {"op": "enter_elevator", "entity_id": "a02-middle_elevator_body"}
  ],
  "bt_skill_mode": "standard_puzzle",
  "constraints": ["avoid_fizzler_while_holding"],
  "memory_append": "room1_door_open_proceed_to_middle_elevator"
}
```

Layer 2 (`BehaviorTreeEvaluator`) consumes `subgoal_queue[0]` by pinning `WorkingMemory.target_entity_id` (and optional constraints) and runs at full 10–66 Hz speed until the subgoal succeeds or triggers a new event.

---

## 4. Honest Failure Modes & Guardrails

1. **Small LLMs are bad at 3D geometry:** Never ask a 1B–3B model for coordinates, angles, or whether a wall has line-of-sight. All spatial predicates (`is_reachable`, `has_los`, `fizzler_between`, `can_fling`) must stay deterministic in Layer 1 (C++ / Python).
2. **Goal thrashing / oscillation:** Calling Layer 4 on every minor perturbation causes it to flip subgoals endlessly. Enforce **commitment hysteresis** (minimum tick dwell time + trigger-only invocation).
3. **Episodic memory drift:** Unbounded prose transcripts degrade small models fast. Cap `episodic_memory` to a fixed rolling ledger (`<= 8` structured single-line facts, deduplicated by entity/event key).
4. **Fallback safety:** If Layer 4 is offline, times out, or emits a blocked subgoal, Layer 2 gracefully falls back to its evolved reactive goal-selection subtree—preserving 100% of our 3-layer zero-shot reliability.

---

## 5. When to Build It

Build Layer 4 **after Milestone 5** (once generic BSP discovery + multi-entity `WorkingMemory` are working live on `testchmb_a_02`), as **Milestone 6**:
1. Define the `StrategicPlanner` protocol in `portal_ml/planner/` (with a deterministic rule/oracle baseline + a pluggable local OpenAI-compatible / `llama.cpp` GBNF client).
2. Benchmark 3-layer vs 4-layer (e.g., 1B–3B local model vs zero-LLM baseline) on multi-room real campaign maps (`testchmb_a_02`, `testchmb_a_05`) measuring solve rate, stall recoveries, and LLM calls per episode.
