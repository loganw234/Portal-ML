# Four-Layer Architecture: Adding a Small-LLM Long-Term Goal & Memory Layer

*Design analysis (2026-11-26) and Milestone 6 implementation + controlled ablation (2026-11-27) for the Layer-4 event-driven strategic planner & episodic memory module.*

## 1. Why a 4th Layer Makes Sense Here

The existing 3-layer stack solves single-room and 3-zone causal puzzles cleanly at high speed (`7/7` chambers live in Portal 1 in 1.5–2.2s):

| Layer | Mechanism | Cadence | Responsibility |
|---|---|---|---|
| **Layer 1 — Perception** | Symbolic Blackboard + BSP Scanner | 66 Hz (server) / 10 Hz (step) | Ground-truth entity states, zones, portal transforms, concrete surfaces, fizzlers |
| **Layer 2 — Reactive Reasoning** | Evolved Behavior Tree DSL (`24–33` nodes) | 10 Hz (`< 0.05 ms/tick`) | Local causal execution, portal bridging/re-linking, reactive grab/drop/press |
| **Layer 3 — Motor Control** | Scripted or Neuroevolved MLPs (`44–83` weights) | 10–66 Hz (`< 0.1 ms/tick`) | Continuous body-frame steering, aiming, future flinging / pellet / turret skills |
| **Layer 4 — Strategic Memory (Implemented, M6)** | Event-driven `StrategicPlanner` (symbolic or `0.5B–4B` local LLM, JSON-schema constrained) | **Event-driven (`~1–4 calls/chamber`)** | Multi-stage subgoal sequencing, Blackboard role binding (`role_overrides`), rolling episodic memory, fizzler & stall recovery |

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

## 5. Milestone 6 Implementation & Controlled Ablation (2026-11-27)

Implemented in `portal_ml/planner/` (zero external dependencies; pure Python 3.12 stdlib):
- **`portal_ml/planner/memory.py`** — `EpisodicMemory(max_entries=8)` with `(event, entity_ids)` deduplication.
- **`portal_ml/planner/protocol.py`** — `TriggerEvent` (`episode_start`, `subgoal_completed`, `cube_fizzled`, `exit_opened`, `stall_detected`), `StrategicDirective.validate_against_snapshot()`, compact `build_symbolic_digest()`, and dynamic `build_directive_json_schema(snapshot)` enumerating only live Blackboard entity IDs (plus `null` role masks) per role.
- **`portal_ml/planner/planners.py`** — `NullStrategicPlanner` (3-layer passthrough), `RuleBasedStrategicPlanner` (deterministic symbolic baseline using only Blackboard + `EpisodicMemory`), and `OpenAICompatibleLLMPlanner` (`urllib.request` client for LM Studio / Ollama / `llama-server` / vLLM with `<think>...</think>` stripping, HTTP 400 schema-fallback retry, and graceful fallback on timeout/connection errors).
- **`portal_ml/planner/controller.py`** — `FourLayerAgent` binding Layer-4 directives into `WorkingMemory.role_overrides` underneath the **frozen** 24-node `champions/seed7_decoupled.bt` (zero BT changes, zero retraining).
- **`portal_ml/planner/benchmarks.py`** — Three multi-stage benchmark chambers kept strictly outside `CHAMBERS[1..7]`:
  - **M1 (`Dual-Button Relay`)**: 2 weighted floor buttons both required (`require_all_buttons=True`), 1 local cube + 1 cube across a portalable pit.
  - **M2 (`Fizzler-Gated Decoy Cube`)**: local decoy cube behind a `FizzlerBarrier` (`trigger_portal_cleanser`), real cube across a portalable pit. Tests episodic failure recovery (`cube_fizzled`).
  - **M3 (`Three-Stage Compound Vault`)**: 3 islands, 2 pits, 2 weighted floor buttons + 1 timed pedestal button all required. Tests multi-stage subgoal sequencing + role masking.

### Measured 3-Layer vs 4-Layer Ablation (Deterministic + Live Local LLMs, 2026-11-27)

Evaluated using the **frozen** 24-node `champions/seed7_decoupled.bt` (zero retraining of Layer 2 or Layer 3). Live local LLM runs executed via LM Studio (`127.0.0.1:1234`, 16k context, dynamic affordance-gated `json_schema`, `temperature=0.0`):

| Configuration | Motor Layer | Standard (Ch 1–7) | Multi-Stage (M1 / M2 / M3) | Total Calls (Avg Latency) | Fallbacks / Warnings | Record |
|---|---|---|---|---|---|---|
| **3-layer (`none`)** | Phase 1 Scripted | **7/7** (`16/18/27/10/45/37/28`) | **0/3** (`FAIL / FAIL / FAIL`) | `0` (`0 ms`) | `0 / 0` | `ablation_layer4_2026-11-27.json` |
| **4-layer (`rule`)** | Phase 1 Scripted | **7/7** (`16/18/27/10/45/37/28`) | **3/3** (`69 / 66 / 94` steps) | `22` (`<0.1 ms`) | `0 / 0` | `ablation_layer4_2026-11-27.json` |
| **4-layer (`phi-4-mini-instruct`, 3.8B)** | Phase 1 Scripted | **7/7** (`16/18/27/10/45/37/28`) | **3/3** (`116 / 66 / 91` steps) | `23` (`470 ms/call`) | `0 / 0` | `ablation_layer4_phi4mini_2026-11-27.json` |
| **4-layer (`liquid/lfm2.5-1.2b`, 1.2B)** | Phase 1 Scripted | **7/7** (`16/18/27/10/45/37/28`) | **1/3** (`FAIL / 42 / FAIL` steps) | `35` (`368 ms/call`) | `0 / 0` | `ablation_layer4_lfm25_2026-11-27.json` |
| **3-layer (`none`)** | Phase 2B Neural (`83w+44w`) | **7/7** (`16/18/27/20/67/56/62`) | **0/3** (`FAIL / FAIL / FAIL`) | `0` (`0 ms`) | `0 / 0` | `ablation_layer4_2026-11-27.json` |
| **4-layer (`rule`)** | Phase 2B Neural (`83w+44w`) | **7/7** (`16/18/27/20/67/56/62`) | **3/3** (`87 / 82 / 149` steps) | `22` (`<0.1 ms`) | `0 / 0` | `ablation_layer4_2026-11-27.json` |
| **4-layer (`phi-4-mini-instruct`, 3.8B)** | Phase 2B Neural (`83w+44w`) | **7/7** (`16/18/27/20/67/56/62`) | **3/3** (`81 / 82 / 152` steps) | `22` (`461 ms/call`) | `0 / 0` | `ablation_layer4_phi4mini_2026-11-27.json` |
| **4-layer (`liquid/lfm2.5-1.2b`, 1.2B)** | Phase 2B Neural (`83w+44w`) | **7/7** (`16/18/27/20/67/56/62`) | **1/3** (`FAIL / 65 / FAIL` steps) | `35` (`332 ms/call`) | `0 / 0` | `ablation_layer4_lfm25_2026-11-27.json` |

Key empirical takeaways:
- **Zero regression on single-stage chambers:** All three Layer-4 planners (`rule`, `phi-4-mini-instruct`, `liquid/lfm2.5-1.2b`) preserve **7/7** bit-for-bit identical step counts across Chambers 1–7 with `0` fallbacks and `0` schema warnings.
- **`phi-4-mini-instruct` (3.8B, 2.49 GB) achieves 10/10 scripted AND 10/10 Phase-2B neural zero-shot** (20/20 total), matching the theoretical minimum `22` planner calls on Phase 2B neural (`3 / 3 / 4` calls on `M1 / M2 / M3`) and beating the rule baseline's step count on M3 scripted (`91` vs `94`) and M1 neural (`81` vs `87`). Lowering context from `16384` -> `4096` (`records/ablation_layer4_phi4mini_4k_2026-11-27.json`) saved **1.55 GB VRAM** on the GPU (`6,254 MiB` -> `4,702 MiB`) with unchanged latency (`466–481 ms/call`) and hit the theoretical minimum **22 calls on both scripted (`62 / 66 / 94` steps) and Phase 2B neural (`81 / 82 / 149` steps)** — making `Phi-4-mini-instruct @ 4k` our strong baseline going forward.
- **`liquid/lfm2.5-1.2b` (1.2B, 1.25 GB) is ~28% faster (~332–368 ms/call) at half the VRAM**, passes **7/7 standard + M2** zero-shot, and exposes a clean Stage-2 multi-button state-switching ceiling on M1/M3 (`switches=1` — completes Stage 1 then re-selects the already-pressed button), making it a crisp benchmark student for Milestone 7 LoRA trace distillation.
- **Architectural lesson discovered during live runs:** Exposing raw Layer-2 BT role-masking quirks (`role_overrides` with `exit=null` / `pedestal_button=null`) tripped both small models, whereas switching Layer 4 to a **semantic-action interface** (`place_cube_on_button`, `press_pedestal`, `go_to_exit`) with Layer-1 physical affordance gating (never offering `go_to_exit` when all exits are `CLOSED`) immediately cut latency ~3–5× and lifted `phi-4-mini-instruct` from `4/7 + 0/3` to **`7/7 + 3/3` (20/20)**!

---

## 6. Local Small-Model Guide: Phi-4-mini vs LFM2.5 (Base vs Instruct)

### Base vs Instruct / Reasoning Checkpoints
- **Zero-shot evaluation (start here):** Use an **Instruct** (or **Thinking/Reasoning**) checkpoint rather than a raw pretrained *base* checkpoint. Raw base weights are next-token continuation models without chat/JSON alignment and will fight structured prompts unless heavily few-shot prompted + GBNF-constrained.
- **LoRA specialization (next phase):** Raw *base* or *Instruct* checkpoints are both strong starting points once you distill Portal-ML JSONL traces (`records/*.jsonl`) into `(symbolic_digest -> StrategicDirective)` supervised LoRA pairs.

### Model Comparison for Portal-ML Layer 4

| Model | Params / Arch | Quant VRAM (Q4–Q8) | Typical Directive Latency (~60 tok) | Best Fit in Portal-ML |
|---|---|---|---|---|
| **`Phi-4-mini-instruct`** (3.8B) | Dense Transformer | ~2.5–4.0 GB | ~40–100 ms (desktop RTX GPU) | **Recommended zero-shot baseline:** strongest out-of-the-box symbolic constraint reasoning & JSON adherence; leaves 12+ GB VRAM free on a 16 GB GPU alongside Portal 1. |
| **`LFM2.5-1.2B-Instruct`** (1.2B) | Hybrid Conv + Attention | ~0.8–1.4 GB | ~15–40 ms (desktop RTX GPU) | **Best ultra-low-latency / LoRA student:** built specifically for fast edge agentic loops; ideal target for LoRA fine-tuning on Portal-ML traces. |
| **`LFM2.5-1.2B-Thinking`** / **`Phi-4-mini-reasoning`** | Reasoning variants | ~0.8–4.0 GB | ~120–400 ms (includes `<think>`) | Useful when debugging multi-room puzzles; `OpenAICompatibleLLMPlanner` automatically strips `<think>...</think>` blocks before parsing JSON. |

### Running a Live Local Model (LM Studio / Ollama / `llama-server`)

Start your local OpenAI-compatible server (e.g., LM Studio on port `1234` with `phi-4-mini-instruct` or `lfm2.5-1.2b-instruct` loaded):

```powershell
# 1. Run the controlled 3-layer vs 4-layer (rule + local LLM) ablation on M1-M3 + Chambers 1-7:
python scripts/ablate_layer4.py `
  --llm-url http://127.0.0.1:1234/v1/chat/completions `
  --llm-model phi-4-mini-instruct `
  --out records/ablation_layer4_phi4mini.json

# 2. Run live inside Portal 1 with Layer 4 enabled:
python scripts/replay_live.py --backend source --planner llm `
  --llm-url http://127.0.0.1:1234/v1/chat/completions `
  --llm-model lfm2.5-1.2b-instruct
```
