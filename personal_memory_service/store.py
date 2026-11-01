"""Bounded SQLite store for explicit, local-only profile facts and memories."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any


PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
KEY_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
SECRET_RE = re.compile(
    r"(?:\b(?:password|passwd|api[_ -]?key|access[_ -]?token|secret)\b|"
    r"\bsk-[A-Za-z0-9_-]{12,})", re.IGNORECASE,
)
PROFILE_FIELDS = {
    "preferred_name", "language", "timezone", "response_style",
    "interests", "accessibility_preferences",
}
LIST_FIELDS = {"interests", "accessibility_preferences"}
MEMORY_KINDS = {"preference", "stable_fact", "goal", "routine"}
CONSENTS = {"user_confirmed", "guardian_confirmed"}


class PersonalMemoryError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def default_database_path() -> Path:
    configured = os.environ.get("VQ_PERSONAL_MEMORY_DB")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local/share/vq-platform/memory/personal-memory.sqlite3"


class PersonalMemoryStore:
    """Local-only store; every durable mutation requires explicit consent."""

    def __init__(self, path: str | Path | None = None, *, max_memories=256, clock=time.time):
        self.path = Path(path or default_database_path())
        self.max_memories = int(max_memories)
        self.clock = clock
        if self.max_memories <= 0:
            raise ValueError("max_memories must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        os.chmod(self.path, 0o600)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS profiles (
              profile_id TEXT NOT NULL, field TEXT NOT NULL, value_json TEXT NOT NULL,
              consent TEXT NOT NULL, updated_at REAL NOT NULL,
              PRIMARY KEY(profile_id, field)
            );
            CREATE TABLE IF NOT EXISTS memories (
              memory_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, kind TEXT NOT NULL,
              memory_key TEXT NOT NULL, value_json TEXT NOT NULL, tags_json TEXT NOT NULL,
              source TEXT NOT NULL, confidence REAL NOT NULL, consent TEXT NOT NULL,
              created_at REAL NOT NULL, updated_at REAL NOT NULL, expires_at REAL,
              UNIQUE(profile_id, kind, memory_key)
            );
            CREATE INDEX IF NOT EXISTS memories_profile_updated
              ON memories(profile_id, updated_at DESC);
        """)
        self._db.commit()

    @staticmethod
    def _profile_id(value: Any) -> str:
        if not isinstance(value, str) or not PROFILE_ID_RE.fullmatch(value):
            raise PersonalMemoryError("invalid_profile_id", "invalid profile_id")
        return value

    @staticmethod
    def _consent(value: Any) -> str:
        if value not in CONSENTS:
            raise PersonalMemoryError("consent_required", "explicit user or guardian confirmation is required")
        return value

    @staticmethod
    def _json(value: Any, *, limit=2048) -> str:
        try:
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise PersonalMemoryError("invalid_value", "value must be JSON") from exc
        if len(encoded) > limit:
            raise PersonalMemoryError("value_too_large", "value exceeds the durable-memory limit")
        if SECRET_RE.search(encoded):
            raise PersonalMemoryError("secret_rejected", "secrets cannot be stored in product memory")
        return encoded

    @staticmethod
    def _field_value(field: str, value: Any) -> Any:
        if field not in PROFILE_FIELDS:
            raise PersonalMemoryError("field_not_allowed", "profile field is not allowed")
        if field in LIST_FIELDS:
            if not isinstance(value, list) or len(value) > 16 or any(
                not isinstance(item, str) or not item.strip() or len(item) > 80 for item in value
            ):
                raise PersonalMemoryError("invalid_value", "invalid profile list value")
            return [item.strip() for item in value]
        if not isinstance(value, str) or not value.strip() or len(value) > 240:
            raise PersonalMemoryError("invalid_value", "invalid profile text value")
        return value.strip()

    def profile_get(self, profile_id: str) -> dict:
        profile_id = self._profile_id(profile_id)
        with self._lock:
            rows = self._db.execute(
                "SELECT field,value_json,consent,updated_at FROM profiles WHERE profile_id=?",
                (profile_id,),
            ).fetchall()
        return {"profile_id": profile_id, "fields": {
            row["field"]: {"value": json.loads(row["value_json"]), "consent": row["consent"],
                           "updated_at": row["updated_at"]} for row in rows
        }}

    def profile_set(self, profile_id: str, field: str, value: Any, *, consent: str) -> dict:
        profile_id, consent = self._profile_id(profile_id), self._consent(consent)
        value = self._field_value(field, value)
        encoded, now = self._json(value, limit=1024), self.clock()
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO profiles VALUES(?,?,?,?,?) ON CONFLICT(profile_id,field) DO UPDATE SET "
                "value_json=excluded.value_json,consent=excluded.consent,updated_at=excluded.updated_at",
                (profile_id, field, encoded, consent, now),
            )
        return self.profile_get(profile_id)

    def profile_forget(self, profile_id: str, field: str) -> dict:
        profile_id = self._profile_id(profile_id)
        if field not in PROFILE_FIELDS:
            raise PersonalMemoryError("field_not_allowed", "profile field is not allowed")
        with self._lock, self._db:
            removed = self._db.execute(
                "DELETE FROM profiles WHERE profile_id=? AND field=?", (profile_id, field)
            ).rowcount > 0
        return {"profile_id": profile_id, "field": field, "forgotten": removed}

    def memory_put(self, profile_id: str, kind: str, key: str, value: Any, *, source: str,
                   consent: str, confidence=1.0, tags=None, ttl_seconds=None) -> dict:
        profile_id, consent = self._profile_id(profile_id), self._consent(consent)
        if kind not in MEMORY_KINDS or not isinstance(key, str) or not KEY_RE.fullmatch(key):
            raise PersonalMemoryError("invalid_memory", "invalid memory kind or key")
        if not isinstance(source, str) or not source.strip() or len(source) > 64:
            raise PersonalMemoryError("invalid_source", "invalid memory source")
        if not isinstance(confidence, (int, float)) or not .5 <= float(confidence) <= 1:
            raise PersonalMemoryError("invalid_confidence", "durable memory confidence must be 0.5 to 1.0")
        tags = [] if tags is None else tags
        if not isinstance(tags, list) or len(tags) > 12 or any(
            not isinstance(tag, str) or not tag.strip() or len(tag) > 40 for tag in tags
        ):
            raise PersonalMemoryError("invalid_tags", "invalid memory tags")
        expires_at = None
        if ttl_seconds is not None:
            if not isinstance(ttl_seconds, (int, float)) or not 3600 <= ttl_seconds <= 365 * 86400:
                raise PersonalMemoryError("invalid_ttl", "durable memory TTL is outside bounds")
            expires_at = self.clock() + float(ttl_seconds)
        encoded, tags_encoded, now = self._json(value), self._json([x.strip() for x in tags], limit=512), self.clock()
        memory_id = "memory-" + uuid.uuid4().hex
        with self._lock, self._db:
            self._purge_locked(now)
            count = self._db.execute("SELECT COUNT(*) FROM memories WHERE profile_id=?", (profile_id,)).fetchone()[0]
            existing = self._db.execute(
                "SELECT memory_id,created_at FROM memories WHERE profile_id=? AND kind=? AND memory_key=?",
                (profile_id, kind, key),
            ).fetchone()
            if existing:
                memory_id, created_at = existing["memory_id"], existing["created_at"]
            else:
                created_at = now
                if count >= self.max_memories:
                    raise PersonalMemoryError("memory_limit", "durable memory limit reached; forget an item first")
            self._db.execute(
                "INSERT OR REPLACE INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (memory_id, profile_id, kind, key, encoded, tags_encoded, source.strip(),
                 float(confidence), consent, created_at, now, expires_at),
            )
        return {"memory_id": memory_id, "profile_id": profile_id, "kind": kind, "key": key,
                "value": value, "tags": tags, "source": source.strip(), "confidence": float(confidence),
                "consent": consent, "created_at": created_at, "updated_at": now, "expires_at": expires_at}

    def _purge_locked(self, now: float) -> None:
        self._db.execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at<=?", (now,))

    def memory_recall(self, profile_id: str, *, query="", limit=6) -> dict:
        profile_id = self._profile_id(profile_id)
        if not isinstance(query, str) or len(query) > 240 or not isinstance(limit, int) or not 1 <= limit <= 12:
            raise PersonalMemoryError("invalid_query", "invalid recall query or limit")
        now = self.clock()
        with self._lock, self._db:
            self._purge_locked(now)
            rows = self._db.execute(
                "SELECT * FROM memories WHERE profile_id=? ORDER BY updated_at DESC LIMIT 256", (profile_id,)
            ).fetchall()
        needle = query.casefold().strip()
        result = []
        for row in rows:
            value = json.loads(row["value_json"]); tags = json.loads(row["tags_json"])
            haystack = " ".join([row["kind"], row["memory_key"], json.dumps(value, ensure_ascii=False), *tags]).casefold()
            if needle and needle not in haystack and not any(char in haystack for char in needle if char.strip()):
                continue
            result.append({key: row[key] for key in (
                "memory_id", "profile_id", "kind", "source", "confidence", "consent",
                "created_at", "updated_at", "expires_at")})
            result[-1].update({"key": row["memory_key"], "value": value, "tags": tags})
            if len(result) >= limit:
                break
        return {"profile_id": profile_id, "query": query, "items": result}

    def memory_forget(self, profile_id: str, memory_id: str) -> dict:
        profile_id = self._profile_id(profile_id)
        if not isinstance(memory_id, str) or not memory_id.startswith("memory-"):
            raise PersonalMemoryError("invalid_memory_id", "invalid memory_id")
        with self._lock, self._db:
            removed = self._db.execute(
                "DELETE FROM memories WHERE profile_id=? AND memory_id=?", (profile_id, memory_id)
            ).rowcount > 0
        return {"profile_id": profile_id, "memory_id": memory_id, "forgotten": removed}

    def reset(self, profile_id: str, *, consent: str) -> dict:
        profile_id, consent = self._profile_id(profile_id), self._consent(consent)
        with self._lock, self._db:
            fields = self._db.execute("DELETE FROM profiles WHERE profile_id=?", (profile_id,)).rowcount
            memories = self._db.execute("DELETE FROM memories WHERE profile_id=?", (profile_id,)).rowcount
        return {"profile_id": profile_id, "reset": True, "removed_fields": fields,
                "removed_memories": memories, "consent": consent}

    def stats(self) -> dict:
        with self._lock, self._db:
            self._purge_locked(self.clock())
            profiles = self._db.execute("SELECT COUNT(DISTINCT profile_id) FROM profiles").fetchone()[0]
            memories = self._db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return {"profiles": profiles, "memories": memories, "persistent": True,
                "local_only": True, "max_memories_per_profile": self.max_memories}

