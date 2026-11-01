"""S7-B: explicit, bounded appliance preferences; not inferred personal traits.

One in-memory value and, only with consent, one small local file. No observer,
model, event subscription, training data, user content or biometric identity.
Expiry is checked at every read/use, without a timer or periodic disk writes.
"""
from __future__ import annotations

import json
import math
import os
import secrets
import stat
import tempfile
import threading
import time
from pathlib import Path

from gateway.speech.text import limit_chat_text, INCOMPLETE_CHAT_REPLY

DEFAULTS = {"speed": 1.0, "reply_length": "standard"}
SESSION_TTL = 30 * 60
REMEMBER_TTL = 7 * 86400
MAX_BYTES = 1024
FIELDS = {"status": set(), "update": {"revision", "speed", "reply_length", "remember"},
          "reset": {"revision"}}


class PreferenceError(RuntimeError):
    pass


def validate(speed, reply_length):
    if type(speed) not in (int, float) or speed not in (1.0, 0.8, 0.6):
        raise ValueError("invalid speed")
    if not isinstance(reply_length, str) or reply_length not in {"standard", "brief"}:
        raise ValueError("invalid reply length")
    return {"speed": float(speed), "reply_length": reply_length}


def fit_reply(text, *, brief, normal_limit):
    """Prefer less speech, never fabricate a truncation error for a long sentence."""
    normal = limit_chat_text(text, normal_limit)
    if brief:
        shorter = limit_chat_text(normal, min(160, normal_limit))
        if shorter != INCOMPLETE_CHAT_REPLY:
            return shorter
    return normal


class InteractionPreferences:
    def __init__(self, root=None, *, wall=time.time, clock=time.monotonic):
        # None is deliberately memory-only, suitable for tests and embedded use.
        self.root = Path(root) if root is not None else None
        self.wall, self.clock = wall, clock
        self.lock = threading.RLock()
        self.values = dict(DEFAULTS)
        self.revision = secrets.token_hex(12)
        self.remember = False
        self.saved_at = self.expires_at = self.deadline = None
        self.source = "default"
        self.last_change = None
        self.storage_error = False
        self._load()

    @property
    def path(self):
        return self.root / "preferences.json"

    @staticmethod
    def _safe(path, *, directory=False):
        info = path.lstat()
        if (stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077
                or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))):
            raise OSError("unsafe preference storage")

    def _load(self):
        if self.root is None or not self.root.exists():
            return
        try:
            self._safe(self.root, directory=True)
            if not self.path.exists() and not self.path.is_symlink():
                return
            self._safe(self.path)
            if self.path.stat().st_size > MAX_BYTES:
                raise ValueError("oversized preferences")
            raw = json.loads(self.path.read_text())
            if (not isinstance(raw, dict) or set(raw) != {"version", "values", "saved_at", "expires_at"}
                    or type(raw["version"]) is not int or raw["version"] != 1
                    or not isinstance(raw["values"], dict) or set(raw["values"]) != set(DEFAULTS)):
                raise ValueError("invalid stored preferences")
            values = validate(**raw["values"])
            start, end = raw["saved_at"], raw["expires_at"]
            if (any(type(t) not in (int, float) or not math.isfinite(t) for t in (start, end))
                    or abs(end - start - REMEMBER_TTL) > .01):
                raise ValueError("invalid expiry")
            if not start - 5 <= self.wall() < end:
                self._erase()
                self.source = "expired"
                return
            self.values, self.saved_at, self.expires_at = values, start, end
            self.deadline = self.clock() + min(REMEMBER_TTL, end - self.wall())
            self.remember, self.source = True, "explicit_local_choice"
        except (OSError, ValueError, TypeError):
            self.storage_error = True  # Do not use or log malformed/private data.

    def _write(self, values, start, end):
        if self.root is None:
            raise PreferenceError("本机保存未配置；可以不勾选记住，仅临时使用")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._safe(self.root, directory=True)
        if self.path.exists() or self.path.is_symlink():
            self._safe(self.path)
        body = json.dumps({"version": 1, "values": values, "saved_at": start,
                           "expires_at": end}, separators=(",", ":"))
        fd, tmp = tempfile.mkstemp(prefix=".preferences-", dir=self.root)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _erase(self):
        if self.root is not None and self.root.exists():
            self._safe(self.root, directory=True)
            if self.path.exists() or self.path.is_symlink():
                self._safe(self.path)
                self.path.unlink()

    def _expire(self):
        if self.expires_at is not None and (self.clock() >= self.deadline
                or self.wall() >= self.expires_at or self.wall() < self.saved_at - 5):
            self.values = dict(DEFAULTS)
            self.remember = False
            self.saved_at = self.expires_at = self.deadline = None
            self.source, self.last_change = "expired", None
            self.revision = secrets.token_hex(12)
            try:
                self._erase()
            except OSError:
                self.storage_error = True

    def snapshot(self):
        with self.lock:
            self._expire()
            return {"revision": self.revision, "values": dict(self.values), "remember": self.remember,
                    "expires_at": self.expires_at, "source": self.source, "scope": "shared_device",
                    "last_change": self.last_change, "storage_error": self.storage_error}

    def call(self, action, data):
        if action not in FIELDS or not isinstance(data, dict) or set(data) != FIELDS[action]:
            raise ValueError("invalid preference request")
        with self.lock:
            self._expire()
            if action == "status":
                return self.snapshot()
            if not isinstance(data["revision"], str) or data["revision"] != self.revision:
                raise PreferenceError("设置已变化或到期，请刷新状态后重试")
            values = dict(DEFAULTS) if action == "reset" else validate(data["speed"], data["reply_length"])
            remember = False if action == "reset" else data["remember"]
            if type(remember) is not bool:
                raise ValueError("explicit remember choice required")
            start = self.wall()
            ttl = REMEMBER_TTL if remember else SESSION_TTL
            try:
                if remember:
                    self._write(values, start, start + ttl)
                else:
                    self._erase()
            except OSError:
                # No success acknowledgement or partial live update on storage failure.
                raise PreferenceError("保存或清除未成功，设置未改变，请检查本机存储") from None
            self.last_change = {"before": dict(self.values), "after": dict(values), "reason": "explicit_choice"}
            self.values, self.remember = values, remember
            self.saved_at = start if action != "reset" else None
            self.expires_at = start + ttl if action != "reset" else None
            self.deadline = self.clock() + ttl if action != "reset" else None
            self.source = "explicit_local_choice" if action != "reset" else "default"
            self.storage_error = False
            self.revision = secrets.token_hex(12)
            return self.snapshot()
