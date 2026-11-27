"""Layer-4 strategic planner implementations: null, rule-based, and local LLM."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from portal_ml.planner.memory import EpisodicMemory
from portal_ml.planner.protocol import (
    StrategicDirective,
    StrategicPlanner,
    TriggerEvent,
    build_directive_json_schema,
    build_symbolic_digest,
    compile_semantic_action,
)
from portal_ml.schema.blackboard import BlackboardSnapshot, EntityKind, EntityState

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are the Layer-4 Strategic Planner for a Portal puzzle agent.\n"
    "Follow these rules in exact priority order:\n"
    "1. If ANY exit is OPEN -> action='go_to_exit', exit_id=<open exit>, others=null.\n"
    "2. Else if ANY floor_button is UNPRESSED and a LOOSE cube exists -> action='place_cube_on_button':\n"
    "   - button_id: pick an UNPRESSED floor_button.\n"
    "   - cube_id: pick a LOOSE cube NOT marked FIZZLED_DO_NOT_USE or STALLED_AVOID.\n"
    "   - pedestal_id: null; exit_id: <exit>.\n"
    "3. Else if ANY pedestal_button is UNPRESSED -> action='press_pedestal', pedestal_id=<unpressed pedestal>, cube_id=null, button_id=null, exit_id=<exit>.\n"
    "4. Else -> action='go_to_exit', exit_id=<exit>, others=null.\n\n"
    "Examples:\n"
    "Input: CUBES: box_a[ON_BUTTON,z=0] | FLOOR_BUTTONS: btn_a[PRESSED,z=0] | EXITS: door_0[OPEN,z=0]\n"
    "Output: {\"action\":\"go_to_exit\",\"cube_id\":null,\"button_id\":null,\"pedestal_id\":null,\"exit_id\":\"door_0\"}\n"
    "Input: CUBES: box_a[ON_BUTTON,z=0]; box_b[ON_BUTTON,z=1] | FLOOR_BUTTONS: btn_a[PRESSED,z=0]; btn_b[PRESSED,z=1] | PEDESTAL_BUTTONS: ped_0[UNPRESSED,z=0] | EXITS: door_0[CLOSED,z=0]\n"
    "Output: {\"action\":\"press_pedestal\",\"cube_id\":null,\"button_id\":null,\"pedestal_id\":\"ped_0\",\"exit_id\":\"door_0\"}\n"
    "Input: CUBES: box_a[ON_BUTTON,z=0]; box_b[LOOSE,z=1] | FLOOR_BUTTONS: btn_a[PRESSED,z=0]; btn_b[UNPRESSED,z=0] | PEDESTAL_BUTTONS: ped_0[UNPRESSED,z=0] | EXITS: door_0[CLOSED,z=0]\n"
    "Output: {\"action\":\"place_cube_on_button\",\"cube_id\":\"box_b\",\"button_id\":\"btn_b\",\"pedestal_id\":null,\"exit_id\":\"door_0\"}\n"
    "Input: CUBES: box_a[LOOSE,z=0,FIZZLED_DO_NOT_USE]; box_b[LOOSE,z=1] | FLOOR_BUTTONS: btn_a[UNPRESSED,z=0] | EXITS: door_0[CLOSED,z=0]\n"
    "Output: {\"action\":\"place_cube_on_button\",\"cube_id\":\"box_b\",\"button_id\":\"btn_a\",\"pedestal_id\":null,\"exit_id\":\"door_0\"}\n"
    "Respond ONLY with JSON matching the schema."
)


class NullStrategicPlanner:
    """3-layer ablation baseline: emits no overrides and leaves Layer 2 untouched."""

    name: str = "none"

    def plan(
        self,
        snapshot: BlackboardSnapshot,
        memory: EpisodicMemory,
        trigger: TriggerEvent,
        current_directive: StrategicDirective | None = None,
    ) -> StrategicDirective:
        return StrategicDirective(
            subgoal="reactive_passthrough",
            role_overrides={},
            memory_note=None,
            min_dwell_steps=10,
        )


class RuleBasedStrategicPlanner:
    """Deterministic symbolic Layer-4 baseline using only Blackboard + EpisodicMemory.

    Holds zero privileged knowledge of map specs, hidden fizzlers, or room layouts:
    it learns fizzler/stall failures strictly from EpisodicMemory entries recorded
    during the episode.
    """

    name: str = "rule"

    def plan(
        self,
        snapshot: BlackboardSnapshot,
        memory: EpisodicMemory,
        trigger: TriggerEvent,
        current_directive: StrategicDirective | None = None,
    ) -> StrategicDirective:
        cubes = snapshot.entities_of_kind(EntityKind.CUBE)
        floor_btns = snapshot.entities_of_kind(EntityKind.FLOOR_BUTTON)
        pedestals = snapshot.entities_of_kind(EntityKind.PEDESTAL_BUTTON)
        exits = snapshot.entities_of_kind(EntityKind.EXIT_DOOR)

        exit_ent = next((e for e in exits if e.active), exits[0] if exits else None)
        exit_id = exit_ent.id if exit_ent is not None else None
        active_cube = next((c for c in cubes if c.active and not c.held), None)

        # Stage A: Exit is open -> keep cube on an already-placed cube (or mask loose cubes)
        # so Layer 2 proceeds straight through the open exit door.
        if exit_ent is not None and exit_ent.active:
            return StrategicDirective(
                subgoal=f"enter open exit {exit_ent.id}",
                role_overrides={
                    "cube": active_cube.id if active_cube is not None else None,
                    "floor_button": None,
                    "pedestal_button": None,
                    "exit": exit_ent.id,
                },
                min_dwell_steps=8,
            )

        # Stage B: Unpressed weighted floor buttons remain -> pair an available
        # non-blacklisted cube with the next inactive floor button, masking pedestals.
        inactive_floor = [b for b in floor_btns if not b.active]
        if inactive_floor and cubes:
            blacklisted = (
                memory.entities_with_event("cube_fizzled")
                | memory.entities_with_event("cube_stalled")
            )
            chosen_btn = self._pick_nearest(snapshot, inactive_floor)
            chosen_cube = self._pick_cube(snapshot, cubes, blacklisted)
            if chosen_cube is not None:
                return StrategicDirective(
                    subgoal=f"place {chosen_cube.id} on {chosen_btn.id}",
                    role_overrides={
                        "cube": chosen_cube.id,
                        "floor_button": chosen_btn.id,
                        "pedestal_button": None,
                        "exit": exit_id,
                    },
                    min_dwell_steps=12,
                )

        # Stage C: All floor buttons satisfied (or none exist) -> press inactive pedestal.
        inactive_peds = [p for p in pedestals if not p.active]
        if inactive_peds:
            chosen_ped = self._pick_nearest(snapshot, inactive_peds)
            return StrategicDirective(
                subgoal=f"press pedestal {chosen_ped.id}",
                role_overrides={
                    "cube": active_cube.id if active_cube is not None else None,
                    "floor_button": None,
                    "pedestal_button": chosen_ped.id,
                    "exit": exit_id,
                },
                min_dwell_steps=8,
            )

        # Default fallback: exit focus
        return StrategicDirective(
            subgoal="default_exit",
            role_overrides={
                "cube": active_cube.id if active_cube is not None else None,
                "floor_button": floor_btns[0].id if floor_btns else None,
                "pedestal_button": pedestals[0].id if pedestals else None,
                "exit": exit_id,
            },
            min_dwell_steps=10,
        )

    @staticmethod
    def _pick_nearest(
        snapshot: BlackboardSnapshot, candidates: list[EntityState]
    ) -> EntityState:
        p = snapshot.player
        return min(
            candidates,
            key=lambda e: (
                0 if e.zone_id == p.zone_id else 1,
                p.pos.distance_xy(e.pos),
                e.id,
            ),
        )

    @staticmethod
    def _pick_cube(
        snapshot: BlackboardSnapshot,
        cubes: list[EntityState],
        blacklisted: set[str],
    ) -> EntityState | None:
        # 1. If already holding a non-blacklisted cube, stay committed to it.
        held_id = snapshot.player.held_entity_id
        if held_id is not None and held_id not in blacklisted:
            held = snapshot.entity_by_id(held_id)
            if held is not None and held.kind is EntityKind.CUBE:
                return held

        # 2. Prefer loose cubes that are neither already holding down a button nor blacklisted.
        unplaced = [c for c in cubes if not c.active]
        preferred = [c for c in unplaced if c.id not in blacklisted]
        pool = preferred if preferred else unplaced
        if not pool:
            return None

        p = snapshot.player
        return min(
            pool,
            key=lambda c: (
                0 if c.zone_id == p.zone_id else 1,
                p.pos.distance_xy(c.pos),
                c.id,
            ),
        )


def strip_reasoning_and_extract_json(raw_text: str) -> dict[str, Any]:
    """Parse JSON from small-LLM output, tolerating <think> blocks and code fences."""
    cleaned = _THINK_BLOCK_RE.sub("", raw_text).strip()
    fence_match = _CODE_FENCE_RE.search(cleaned)
    if fence_match is not None:
        cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: locate the outermost {...} JSON object in the remaining text.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        parsed = json.loads(cleaned[start : end + 1])
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"Could not extract JSON object from LLM response: {raw_text[:160]!r}")


class OpenAICompatibleLLMPlanner:
    """Zero-dependency OpenAI-compatible HTTP client for local small LLMs.

    Works out-of-the-box with LM Studio (``http://127.0.0.1:1234/v1/chat/completions``),
    Ollama (``http://127.0.0.1:11434/v1/chat/completions``), ``llama-server``, and vLLM.
    Enforces dynamic per-snapshot JSON Schema structured output (with graceful fallback
    to ``json_object`` and to a local fallback planner on HTTP/timeout/parse errors).
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234/v1/chat/completions",
        model: str = "phi-4-mini-instruct",
        *,
        api_key: str = "lm-studio",
        timeout_s: float = 5.0,
        max_tokens: int = 256,
        temperature: float = 0.0,
        use_json_schema: bool = True,
        fallback_planner: StrategicPlanner | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_json_schema = use_json_schema
        self.fallback_planner: StrategicPlanner = (
            fallback_planner if fallback_planner is not None else NullStrategicPlanner()
        )
        self.name = f"llm:{model}"

        # Telemetry
        self.calls: int = 0
        self.fallbacks: int = 0
        self.last_error: str | None = None
        self.last_raw_response: str | None = None

    def plan(
        self,
        snapshot: BlackboardSnapshot,
        memory: EpisodicMemory,
        trigger: TriggerEvent,
        current_directive: StrategicDirective | None = None,
    ) -> StrategicDirective:
        self.calls += 1
        self.last_error = None
        digest = build_symbolic_digest(snapshot, memory, trigger, current_directive)
        schema = build_directive_json_schema(snapshot) if self.use_json_schema else None

        try:
            raw_content = self._request_completion(digest, schema)
            self.last_raw_response = raw_content
            payload = strip_reasoning_and_extract_json(raw_content)
            return self._parse_directive(snapshot, payload)
        except Exception as exc:
            self.fallbacks += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            return self.fallback_planner.plan(
                snapshot, memory, trigger, current_directive
            )

    def _request_completion(
        self, user_digest: str, json_schema: dict[str, Any] | None
    ) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_digest},
        ]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }

        try:
            return self._post_json(body)
        except urllib.error.HTTPError as err:
            # Some OpenAI-compatible servers reject json_schema with HTTP 400;
            # retry once with plain JSON instructions.
            if err.code == 400 and "response_format" in body:
                body.pop("response_format", None)
                return self._post_json(body)
            raise

    def _post_json(self, body: dict[str, Any]) -> str:
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url, data=data, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))

        choices = resp_data.get("choices") or []
        if not choices:
            raise ValueError("LLM response contained no choices")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response choice had empty message content")
        return content

    @staticmethod
    def _parse_directive(
        snapshot: BlackboardSnapshot, payload: dict[str, Any]
    ) -> StrategicDirective:
        if "action" in payload:
            return compile_semantic_action(
                snapshot,
                action=str(payload["action"]),
                cube_id=(
                    None if payload.get("cube_id") is None else str(payload["cube_id"])
                ),
                button_id=(
                    None
                    if payload.get("button_id") is None
                    else str(payload["button_id"])
                ),
                pedestal_id=(
                    None
                    if payload.get("pedestal_id") is None
                    else str(payload["pedestal_id"])
                ),
                exit_id=(
                    None if payload.get("exit_id") is None else str(payload["exit_id"])
                ),
            )
        raw_overrides = payload.get("role_overrides", {})
        if not isinstance(raw_overrides, dict):
            raise ValueError("Directive must contain 'action' or 'role_overrides' object")
        overrides: dict[str, str | None] = {}
        for k, v in raw_overrides.items():
            overrides[str(k)] = None if v is None else str(v)
        return StrategicDirective(
            subgoal=str(payload.get("subgoal", "llm_directive")),
            role_overrides=overrides,
            memory_note=(
                None
                if payload.get("memory_note") is None
                else str(payload.get("memory_note"))
            ),
            min_dwell_steps=int(payload.get("min_dwell_steps", 10)),
        )
