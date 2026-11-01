"""Bounded in-memory session lifecycle shared by interaction and conversation pipelines."""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from governance import PrivacyContext
from .interaction import Affect, InteractionEvent, InteractionOrchestrator


@dataclass
class SessionRecord:
    owner: str
    created_at: float
    expires_at: float
    interaction: InteractionOrchestrator = field(default_factory=InteractionOrchestrator)
    trace_id: str | None = None
    turns: list[dict[str, str]] = field(default_factory=list)
    privacy: PrivacyContext = field(default_factory=PrivacyContext)


class SessionStore:
    """Thread-safe, non-persistent, LRU-bounded session store."""

    def __init__(self, *, ttl_seconds=12 * 60 * 60, max_sessions=32, max_turns=4, clock=time.monotonic):
        if ttl_seconds <= 0 or max_sessions <= 0 or max_turns <= 0:
            raise ValueError("session bounds must be positive")
        self.ttl_seconds = float(ttl_seconds)
        self.max_sessions = int(max_sessions)
        self.max_turns = int(max_turns)
        self.clock = clock
        self._records: OrderedDict[str, SessionRecord] = OrderedDict()
        self._lock = threading.Lock()

    def _purge(self, now: float) -> None:
        for owner in [key for key, value in self._records.items() if value.expires_at <= now]:
            self._records.pop(owner, None)

    def _record(self, owner: str) -> SessionRecord:
        if not owner:
            raise ValueError("session owner is required")
        now = self.clock()
        self._purge(now)
        record = self._records.get(owner)
        if record is None:
            while len(self._records) >= self.max_sessions:
                self._records.popitem(last=False)
            record = SessionRecord(owner, now, now + self.ttl_seconds)
            self._records[owner] = record
        else:
            record.expires_at = now + self.ttl_seconds
            self._records.move_to_end(owner)
        return record

    def context(self, owner: str) -> list[dict[str, str]]:
        with self._lock:
            return [dict(item) for item in self._record(owner).turns]

    def append_turn(self, owner: str, user_text: str, buddy_text: str) -> None:
        values = (user_text.strip(), buddy_text.strip())
        if not all(values) or any(len(value) > 800 for value in values):
            raise ValueError("session turn is empty or too long")
        with self._lock:
            record = self._record(owner)
            record.turns.append({"role": "user", "text": values[0][:300]})
            record.turns.append({"role": "buddy", "text": values[1][:300]})
            record.turns[:] = record.turns[-self.max_turns * 2:]

    def set_privacy(self, owner: str, privacy: PrivacyContext) -> None:
        if not isinstance(privacy, PrivacyContext):
            raise ValueError("invalid privacy context")
        with self._lock:
            self._record(owner).privacy = privacy

    def privacy(self, owner: str) -> PrivacyContext:
        with self._lock:
            return self._record(owner).privacy

    def transition(self, owner: str, event: InteractionEvent, affect: Affect | None = None):
        with self._lock:
            record = self._record(owner)
            if record.trace_id is None:
                record.trace_id = "interaction-" + uuid.uuid4().hex
            trace_id = record.trace_id
            if affect is not None:
                record.interaction.update_affect(affect)
            intent = record.interaction.transition(event)
            snapshot = record.interaction.snapshot()
            if event in {InteractionEvent.SPEECH_FINISHED, InteractionEvent.RESET}:
                record.trace_id = None
            return trace_id, intent, snapshot

    def ensure_trace(self, owner: str, *, renew_if_idle: bool = False) -> str:
        with self._lock:
            record = self._record(owner)
            if record.trace_id is None or (renew_if_idle and record.interaction.phase.value == "idle"):
                record.trace_id = "interaction-" + uuid.uuid4().hex
            return record.trace_id

    def update_affect(self, owner: str, affect: Affect) -> None:
        if not isinstance(affect, Affect):
            raise ValueError("invalid affect")
        with self._lock:
            self._record(owner).interaction.update_affect(affect)

    def reset(self, owner: str) -> bool:
        with self._lock:
            return self._records.pop(owner, None) is not None

    def stats(self) -> dict:
        with self._lock:
            self._purge(self.clock())
            return {"active_sessions": len(self._records), "max_sessions": self.max_sessions,
                    "ttl_seconds": self.ttl_seconds, "max_turns": self.max_turns}
