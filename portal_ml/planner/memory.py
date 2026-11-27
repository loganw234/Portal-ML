"""Bounded, deduplicated episodic memory for Layer-4 strategic planning.

Stores concise, structured causal discoveries across ticks (e.g. fizzler
vaporizations, activated buttons, stall events) with a strict capacity cap so
small-LLM prompt context stays tiny and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EpisodicEntry:
    """One atomic causal fact recorded during an episode."""

    tick: int
    event: str
    entity_ids: tuple[str, ...]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "event": self.event,
            "entity_ids": list(self.entity_ids),
            "note": self.note,
        }

    def format_line(self) -> str:
        ids = ",".join(self.entity_ids) if self.entity_ids else "-"
        return f"- [t={self.tick}] {self.event} ({ids}): {self.note}"


class EpisodicMemory:
    """Fixed-capacity FIFO episodic log with key-based deduplication.

    Deduplication key is ``(event, entity_ids)`` so repeated identical triggers
    refresh the timestamp/note without evicting distinct causal facts.
    """

    def __init__(self, max_entries: int = 8) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        self._entries: list[EpisodicEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[EpisodicEntry, ...]:
        return tuple(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def record(
        self,
        tick: int,
        event: str,
        entity_ids: tuple[str, ...] | list[str] = (),
        note: str = "",
    ) -> EpisodicEntry:
        norm_ids = tuple(str(e) for e in entity_ids)
        clean_note = " ".join(str(note).split())[:120]
        entry = EpisodicEntry(
            tick=int(tick),
            event=str(event),
            entity_ids=norm_ids,
            note=clean_note,
        )
        # Deduplicate on (event, entity_ids), moving updated entry to most recent
        self._entries = [
            existing
            for existing in self._entries
            if not (existing.event == entry.event and existing.entity_ids == entry.entity_ids)
        ]
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]
        return entry

    def has_event(self, event: str, entity_id: str | None = None) -> bool:
        for e in self._entries:
            if e.event != event:
                continue
            if entity_id is None or entity_id in e.entity_ids:
                return True
        return False

    def entities_with_event(self, event: str) -> set[str]:
        out: set[str] = set()
        for e in self._entries:
            if e.event == event:
                out.update(e.entity_ids)
        return out

    def format_for_prompt(self) -> str:
        if not self._entries:
            return "(none)"
        return "\n".join(e.format_line() for e in self._entries)

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._entries]
