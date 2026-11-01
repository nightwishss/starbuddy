"""Thread-safe, bounded short-term conversation state.

This store deliberately keeps no durable profile data.  A service restart is a
privacy boundary and clears every session.  Durable, consented user profiles
belong to a separate memory service.
"""

from __future__ import annotations

import copy
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable


SESSION_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
MEMORY_KEY_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
INPUT_TYPES = {"text", "touch", "voice", "event", "api"}


class ConversationError(Exception):
    """Stable protocol error returned to clients."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class MemoryItem:
    key: str
    value: Any
    source: str
    confidence: float
    created_at: float
    expires_at: float


@dataclass
class ActiveTurn:
    turn_id: str
    user_text: str
    input_type: str
    started_at: float
    expires_at: float


@dataclass
class ConversationRecord:
    session_id: str
    created_at: float
    updated_at: float
    expires_at: float
    privacy: str = "local-private"
    revision: int = 1
    messages: list[dict[str, Any]] = field(default_factory=list)
    memory: dict[str, MemoryItem] = field(default_factory=dict)
    active_turn: ActiveTurn | None = None
    last_turn: dict[str, Any] | None = None


class ConversationStore:
    """In-memory LRU store with atomic turns and structured working memory."""

    def __init__(
        self, *, ttl_seconds: float = 2 * 60 * 60, max_sessions: int = 64,
        max_turns: int = 8, max_context_chars: int = 6000,
        max_memory_items: int = 16, max_text_chars: int = 1200,
        turn_timeout_seconds: float = 90,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ):
        bounds = (ttl_seconds, max_sessions, max_turns, max_context_chars,
                  max_memory_items, max_text_chars, turn_timeout_seconds)
        if any(value <= 0 for value in bounds):
            raise ValueError("conversation bounds must be positive")
        self.ttl_seconds = float(ttl_seconds)
        self.max_sessions = int(max_sessions)
        self.max_turns = int(max_turns)
        self.max_context_chars = int(max_context_chars)
        self.max_memory_items = int(max_memory_items)
        self.max_text_chars = int(max_text_chars)
        self.turn_timeout_seconds = min(float(turn_timeout_seconds), self.ttl_seconds)
        self.clock = clock
        self.wall_clock = wall_clock
        self._records: OrderedDict[str, ConversationRecord] = OrderedDict()
        self._condition = threading.Condition(threading.RLock())

    @staticmethod
    def _session_id(value: Any) -> str:
        if not isinstance(value, str) or not SESSION_RE.fullmatch(value):
            raise ConversationError("invalid_session_id", "invalid session_id")
        return value

    def _text(self, value: Any, field_name: str) -> str:
        if not isinstance(value, str):
            raise ConversationError("invalid_text", f"{field_name} must be text")
        value = value.strip()
        if not value or len(value) > self.max_text_chars:
            raise ConversationError("invalid_text", f"invalid {field_name} length")
        return value

    def _purge(self, now: float) -> None:
        expired = [key for key, record in self._records.items() if record.expires_at <= now]
        for key in expired:
            self._records.pop(key, None)

    def _get(self, session_id: str, *, create: bool = True) -> ConversationRecord:
        session_id = self._session_id(session_id)
        now = self.clock()
        self._purge(now)
        record = self._records.get(session_id)
        if record is None:
            if not create:
                raise ConversationError("session_not_found", "session not found")
            while len(self._records) >= self.max_sessions:
                self._records.popitem(last=False)
            record = ConversationRecord(
                session_id=session_id, created_at=self.wall_clock(),
                updated_at=self.wall_clock(), expires_at=now + self.ttl_seconds,
            )
            self._records[session_id] = record
        else:
            record.expires_at = now + self.ttl_seconds
            self._records.move_to_end(session_id)
        self._expire_active_turn(record, now)
        self._purge_memory(record, now)
        return record

    def _expire_active_turn(self, record: ConversationRecord, now: float) -> None:
        active = record.active_turn
        if active is None or active.expires_at > now:
            return
        record.active_turn = None
        record.last_turn = {
            "turn_id": active.turn_id, "status": "failed",
            "finished_at": self.wall_clock(), "reason": "turn_timeout",
        }
        self._changed(record)

    def _purge_memory(self, record: ConversationRecord, now: float) -> None:
        for key in [key for key, item in record.memory.items() if item.expires_at <= now]:
            record.memory.pop(key, None)

    def _changed(self, record: ConversationRecord) -> None:
        record.revision += 1
        record.updated_at = self.wall_clock()
        self._condition.notify_all()

    def ensure(self, session_id: str, *, privacy: str = "local-private") -> dict:
        if privacy not in {"local-private", "local-sensitive"}:
            raise ConversationError("invalid_privacy", "invalid privacy label")
        with self._condition:
            session_id = self._session_id(session_id)
            self._purge(self.clock())
            existed = session_id in self._records
            record = self._get(session_id)
            if not existed and record.privacy != privacy:
                record.privacy = privacy
            return self._snapshot(record)

    def start_turn(self, session_id: str, user_text: str, *, input_type: str = "text") -> dict:
        user_text = self._text(user_text, "user_text")
        if input_type not in INPUT_TYPES:
            raise ConversationError("invalid_input_type", "invalid input_type")
        with self._condition:
            record = self._get(session_id)
            if record.active_turn is not None:
                raise ConversationError("turn_in_progress", "a turn is already active")
            record.active_turn = ActiveTurn(
                turn_id="turn-" + uuid.uuid4().hex,
                user_text=user_text, input_type=input_type, started_at=self.wall_clock(),
                expires_at=self.clock() + self.turn_timeout_seconds,
            )
            self._changed(record)
            return self._snapshot(record)

    def complete_turn(self, session_id: str, turn_id: str, assistant_text: str,
                      *, backend: str = "unknown") -> dict:
        assistant_text = self._text(assistant_text, "assistant_text")
        if not isinstance(backend, str) or not backend.strip() or len(backend) > 64:
            raise ConversationError("invalid_backend", "invalid backend")
        with self._condition:
            record = self._get(session_id, create=False)
            active = self._active(record, turn_id)
            timestamp = self.wall_clock()
            record.messages.extend([
                {"role": "user", "text": active.user_text, "turn_id": active.turn_id,
                 "input_type": active.input_type, "timestamp": active.started_at},
                {"role": "assistant", "text": assistant_text, "turn_id": active.turn_id,
                 "backend": backend.strip(), "timestamp": timestamp},
            ])
            record.active_turn = None
            record.last_turn = {"turn_id": active.turn_id, "status": "completed",
                                "finished_at": timestamp, "backend": backend.strip()}
            self._trim_messages(record)
            self._changed(record)
            return self._snapshot(record)

    def end_turn(self, session_id: str, turn_id: str, *, status: str,
                 reason: str | None = None) -> dict:
        if status not in {"failed", "cancelled"}:
            raise ConversationError("invalid_status", "invalid terminal status")
        if reason is not None and (not isinstance(reason, str) or len(reason) > 160):
            raise ConversationError("invalid_reason", "invalid reason")
        with self._condition:
            record = self._get(session_id, create=False)
            active = self._active(record, turn_id)
            record.active_turn = None
            record.last_turn = {"turn_id": active.turn_id, "status": status,
                                "finished_at": self.wall_clock(), "reason": reason}
            self._changed(record)
            return self._snapshot(record)

    def fail_turn(self, session_id: str, turn_id: str, *, reason: str | None = None) -> dict:
        return self.end_turn(session_id, turn_id, status="failed", reason=reason)

    def cancel_turn(self, session_id: str, turn_id: str, *, reason: str | None = None) -> dict:
        return self.end_turn(session_id, turn_id, status="cancelled", reason=reason)

    @staticmethod
    def _active(record: ConversationRecord, turn_id: str) -> ActiveTurn:
        if not isinstance(turn_id, str) or record.active_turn is None:
            raise ConversationError("turn_not_found", "active turn not found")
        if record.active_turn.turn_id != turn_id:
            raise ConversationError("turn_mismatch", "turn_id does not own active turn")
        return record.active_turn

    def _trim_messages(self, record: ConversationRecord) -> None:
        record.messages[:] = record.messages[-self.max_turns * 2:]
        total = sum(len(message["text"]) for message in record.messages)
        while len(record.messages) > 2 and total > self.max_context_chars:
            removed = record.messages[:2]
            del record.messages[:2]
            total -= sum(len(message["text"]) for message in removed)

    def remember(self, session_id: str, key: str, value: Any, *, source: str,
                 confidence: float = 1.0, ttl_seconds: float | None = None) -> dict:
        if not isinstance(key, str) or not MEMORY_KEY_RE.fullmatch(key):
            raise ConversationError("invalid_memory_key", "invalid memory key")
        if not isinstance(source, str) or not source.strip() or len(source) > 64:
            raise ConversationError("invalid_memory_source", "invalid memory source")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ConversationError("invalid_confidence", "confidence must be between 0 and 1")
        try:
            import json
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ConversationError("invalid_memory_value", "memory value must be JSON") from exc
        if len(encoded) > 1024:
            raise ConversationError("memory_value_too_large", "memory value exceeds 1024 characters")
        duration = self.ttl_seconds if ttl_seconds is None else float(ttl_seconds)
        if duration <= 0 or duration > self.ttl_seconds:
            raise ConversationError("invalid_memory_ttl", "memory ttl is outside session bounds")
        with self._condition:
            record = self._get(session_id)
            if key not in record.memory and len(record.memory) >= self.max_memory_items:
                oldest = min(record.memory.values(), key=lambda item: item.created_at)
                record.memory.pop(oldest.key, None)
            record.memory[key] = MemoryItem(
                key=key, value=copy.deepcopy(value), source=source.strip(),
                confidence=float(confidence), created_at=self.wall_clock(),
                expires_at=self.clock() + duration,
            )
            self._changed(record)
            return self._snapshot(record)

    def forget(self, session_id: str, key: str) -> dict:
        with self._condition:
            record = self._get(session_id, create=False)
            removed = record.memory.pop(key, None) is not None
            if removed:
                self._changed(record)
            result = self._snapshot(record)
            result["forgotten"] = removed
            return result

    def context(self, session_id: str) -> dict:
        with self._condition:
            return self._snapshot(self._get(session_id))

    def reset(self, session_id: str) -> dict:
        session_id = self._session_id(session_id)
        with self._condition:
            removed = self._records.pop(session_id, None) is not None
            self._condition.notify_all()
            return {"session_id": session_id, "reset": removed}

    def wait(self, session_id: str, after_revision: int, timeout: float = 20.0) -> dict:
        if not isinstance(after_revision, int) or after_revision < 0:
            raise ConversationError("invalid_revision", "invalid revision")
        timeout = min(max(float(timeout), 0.0), 30.0)
        deadline = self.clock() + timeout
        with self._condition:
            record = self._get(session_id)
            while record.revision <= after_revision:
                remaining = deadline - self.clock()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
                record = self._get(session_id)
            result = self._snapshot(record)
            result["changed"] = record.revision > after_revision
            return result

    def stats(self) -> dict:
        with self._condition:
            self._purge(self.clock())
            return {
                "active_sessions": len(self._records), "max_sessions": self.max_sessions,
                "ttl_seconds": self.ttl_seconds, "max_turns": self.max_turns,
                "max_context_chars": self.max_context_chars,
                "turn_timeout_seconds": self.turn_timeout_seconds,
                "active_turns": sum(record.active_turn is not None for record in self._records.values()),
                "persistent": False,
            }

    def _snapshot(self, record: ConversationRecord) -> dict:
        now = self.clock()
        self._purge_memory(record, now)
        active = None
        if record.active_turn is not None:
            active = {
                "turn_id": record.active_turn.turn_id,
                "user_text": record.active_turn.user_text,
                "input_type": record.active_turn.input_type,
                "started_at": record.active_turn.started_at,
                "expires_in_seconds": max(0, round(record.active_turn.expires_at - now, 3)),
            }
        memory = {
            key: {"value": copy.deepcopy(item.value), "source": item.source,
                  "confidence": item.confidence,
                  "expires_in_seconds": max(0, round(item.expires_at - now, 3))}
            for key, item in sorted(record.memory.items())
        }
        return {
            "session_id": record.session_id, "privacy": record.privacy,
            "revision": record.revision, "created_at": record.created_at,
            "updated_at": record.updated_at, "expires_in_seconds": max(
                0, round(record.expires_at - now, 3)),
            "messages": copy.deepcopy(record.messages), "memory": memory,
            "active_turn": active, "last_turn": copy.deepcopy(record.last_turn),
        }
