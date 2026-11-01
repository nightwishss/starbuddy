"""Lease-based cross-process admission with bounded capacity and expiry."""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import stat
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


OWNER = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
LEASE = re.compile(r"^lease_[A-Za-z0-9_-]{20,64}$")
DEFAULT_CAPACITY = {"npu_compute": 100, "cpu_compute": 100, "ram_mb": 12000,
                    "camera_stream": 1, "audio_output": 1, "local_llm": 1}
LLM_BACKGROUND_IDLE_SECONDS = 10.0
LLM_BACKGROUND_NPU_COMPUTE = 70  # qwen3-4b-instruct-2507 in models.json
MAX_LLM_MARKER_BYTES = 2048


class SchedulerError(ValueError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.details = code, details or {}


@dataclass
class LeaseRecord:
    lease_id: str
    owner: str
    resources: dict
    priority: int
    purpose: str
    created_at: float
    expires_at: float

    def public(self):
        return asdict(self)


class ResourceScheduler:
    def __init__(self, capacity=None, *, clock=time.monotonic, state_path=None, foreground_npu=False):
        self.capacity = dict(capacity or DEFAULT_CAPACITY)
        self.capacity["local_llm"] = 1
        self.clock = clock
        self.foreground_npu = foreground_npu
        self.leases = {}
        self.lock = threading.Lock()
        self._llm = None
        self._last_foreground_at = None
        self._llm_idle_since = self.clock()
        self.state_path = Path(state_path) if state_path is not None else None
        self._marker_record = None
        self._marker_identity = None
        self._llm_state_error = None
        if self.state_path is not None:
            try:
                record, identity = self._read_marker()
                self._marker_record, self._marker_identity = record, identity
                lease = LeaseRecord(record["token"], record["owner"], record["resources"],
                                    90 if record["kind"] == "foreground" else 10,
                                    "llm." + record["kind"], record["created_at"], record["expires_at"])
                self.leases[lease.lease_id] = lease
                self._llm = {"token": lease.lease_id, "owner": lease.owner,
                             "kind": record["kind"], "state": "quarantined"}
            except FileNotFoundError:
                pass
            except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
                # Never replace or delete a marker whose ownership or contents
                # cannot be established. An old native request may still run.
                self._llm_state_error = "llm_marker_unsafe"

    def _llm_resources(self, kind):
        return ({"local_llm": 1, "npu_compute": LLM_BACKGROUND_NPU_COMPUTE}
                if kind == "background" or self.foreground_npu else {"local_llm": 1})

    def _read_marker(self):
        descriptor = os.open(self.state_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            metadata = os.fstat(handle.fileno())
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                    or stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_nlink != 1
                    or metadata.st_size > MAX_LLM_MARKER_BYTES):
                raise ValueError("unsafe language lane marker")
            raw = handle.read(MAX_LLM_MARKER_BYTES + 1)
        if len(raw) > MAX_LLM_MARKER_BYTES:
            raise ValueError("oversized language lane marker")
        try:
            record = json.loads(raw)
        except RecursionError as exc:
            raise ValueError("invalid language lane marker nesting") from exc
        if (not isinstance(record, dict) or set(record) != {
                "schema_version", "token", "owner", "kind", "resources", "created_at", "expires_at"}
                or record["schema_version"] != 1
                or not isinstance(record["token"], str) or not LEASE.fullmatch(record["token"])
                or not isinstance(record["owner"], str) or not OWNER.fullmatch(record["owner"])
                or record["kind"] not in ("foreground", "background")
                or record["resources"] not in (
                    {"local_llm": 1, "npu_compute": LLM_BACKGROUND_NPU_COMPUTE},
                    {"local_llm": 1} if record["kind"] == "foreground" else {})):
            raise ValueError("invalid language lane marker")
        for key in ("created_at", "expires_at"):
            value = record[key]
            try:
                valid_time = (not isinstance(value, bool) and isinstance(value, (int, float))
                              and math.isfinite(value) and value >= 0)
            except OverflowError:
                valid_time = False
            if not valid_time:
                raise ValueError("invalid language lane marker time")
        if not 1 <= record["expires_at"] - record["created_at"] <= 300:
            raise ValueError("invalid language lane marker lifetime")
        self._resources(record["resources"])
        return record, (metadata.st_dev, metadata.st_ino)

    def _marker_directory(self):
        descriptor = os.open(self.state_path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o022:
            os.close(descriptor)
            raise ValueError("unsafe language lane marker directory")
        return descriptor

    def _write_marker(self, lease, kind):
        if self.state_path is None:
            return
        record = {"schema_version": 1, "token": lease.lease_id, "owner": lease.owner,
                  "kind": kind, "resources": lease.resources,
                  "created_at": lease.created_at, "expires_at": lease.expires_at}
        raw = json.dumps(record, separators=(",", ":")).encode("utf-8")
        if len(raw) > MAX_LLM_MARKER_BYTES:
            raise ValueError("oversized language lane marker")
        directory = self._marker_directory()
        temporary = None
        try:
            descriptor, temporary = tempfile.mkstemp(prefix=".vq-llm-lane-", dir=self.state_path.parent)
            with os.fdopen(descriptor, "wb") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
                metadata = os.fstat(handle.fileno())
            # Publication must not overwrite a marker created by another owner
            # or an older service. Linking a prepared file is atomic and fails
            # if the destination already exists.
            os.link(temporary, self.state_path.name, dst_dir_fd=directory, follow_symlinks=False)
            os.fsync(directory)
            self._marker_record = record
            self._marker_identity = (metadata.st_dev, metadata.st_ino)
        finally:
            if temporary is not None:
                os.unlink(temporary)
            os.close(directory)

    def _remove_marker(self):
        if self.state_path is None:
            return
        record, identity = self._read_marker()
        if record != self._marker_record or identity != self._marker_identity:
            raise ValueError("language lane marker changed")
        directory = self._marker_directory()
        try:
            os.unlink(self.state_path.name, dir_fd=directory)
            os.fsync(directory)
        finally:
            os.close(directory)
        self._marker_record = self._marker_identity = None

    def _purge(self):
        now = self.clock()
        managed = self._llm["token"] if self._llm is not None else None
        if managed is not None and self.leases[managed].expires_at <= now:
            # A client deadline or lost process is not native generation
            # termination. Keep the lane occupied until an acknowledgement.
            self._llm["state"] = "quarantined"
        for key in [key for key, lease in self.leases.items()
                    if lease.expires_at <= now and key != managed]:
            self.leases.pop(key, None)

    def _resources(self, value):
        if not isinstance(value, dict) or not value:
            raise SchedulerError("invalid_resources", "resources must be non-empty")
        result = {}
        for key, amount in value.items():
            if key not in self.capacity or isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
                raise SchedulerError("invalid_resources", "invalid resource request")
            if amount > self.capacity[key]:
                raise SchedulerError("request_too_large", "request exceeds capacity", {key: self.capacity[key]})
            result[key] = amount
        return result

    def acquire(self, *, owner, resources, purpose, priority=50, ttl_seconds=30):
        if not isinstance(owner, str) or not OWNER.fullmatch(owner):
            raise SchedulerError("invalid_owner", "invalid owner")
        if not isinstance(purpose, str) or not OWNER.fullmatch(purpose):
            raise SchedulerError("invalid_purpose", "invalid purpose")
        if not isinstance(priority, int) or not 0 <= priority <= 100:
            raise SchedulerError("invalid_priority", "priority must be 0..100")
        if not isinstance(ttl_seconds, (int, float)) or not 1 <= ttl_seconds <= 300:
            raise SchedulerError("invalid_ttl", "ttl must be 1..300 seconds")
        request = self._resources(resources)
        with self.lock:
            self._purge()
            return self._acquire_locked(owner, request, purpose, priority, ttl_seconds).public()

    def _acquire_locked(self, owner, request, purpose, priority, ttl_seconds):
        used = self._used_locked()
        unavailable = {key: {"requested": amount, "available": self.capacity[key] - used[key]}
                       for key, amount in request.items()
                       if amount > self.capacity[key] - used[key]}
        if unavailable:
            raise SchedulerError("resource_busy", "resources unavailable", unavailable)
        now = self.clock()
        lease = LeaseRecord("lease_" + secrets.token_urlsafe(18), owner, request,
                            priority, purpose, now, now + ttl_seconds)
        self.leases[lease.lease_id] = lease
        return lease

    def renew(self, *, lease_id, owner, ttl_seconds=30):
        if not isinstance(ttl_seconds, (int, float)) or not 1 <= ttl_seconds <= 300:
            raise SchedulerError("invalid_ttl", "ttl must be 1..300 seconds")
        with self.lock:
            self._purge()
            lease = self.leases.get(lease_id)
            if lease is None:
                raise SchedulerError("lease_not_found", "lease not found")
            if lease.owner != owner:
                raise SchedulerError("permission_denied", "lease owner mismatch")
            if self._llm is not None and lease_id == self._llm["token"]:
                raise SchedulerError("llm_lane_managed", "use llm.finish for the language lane")
            lease.expires_at = self.clock() + ttl_seconds
            return lease.public()

    def release(self, *, lease_id, owner):
        with self.lock:
            self._purge()
            lease = self.leases.get(lease_id)
            if lease is None:
                return {"lease_id": lease_id, "released": False}
            if lease.owner != owner:
                raise SchedulerError("permission_denied", "lease owner mismatch")
            if self._llm is not None and lease_id == self._llm["token"]:
                raise SchedulerError("llm_lane_managed", "use llm.finish for the language lane")
            self.leases.pop(lease_id)
            return {"lease_id": lease_id, "released": True}

    def llm_enter(self, *, owner, kind="foreground", ttl_seconds=30):
        if not isinstance(owner, str) or not OWNER.fullmatch(owner):
            raise SchedulerError("invalid_owner", "invalid owner")
        if not isinstance(kind, str) or kind not in {"foreground", "background"}:
            raise SchedulerError("invalid_kind", "kind must be foreground or background")
        if (isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float))
                or not 1 <= ttl_seconds <= 300):
            raise SchedulerError("invalid_ttl", "ttl must be 1..300 seconds")
        with self.lock:
            self._purge()
            now = self.clock()
            if kind == "foreground":
                # Even a rejected foreground request postpones background work.
                self._last_foreground_at = now
            if self._llm_state_error is not None:
                raise SchedulerError(self._llm_state_error, "local language lane persistence is unavailable")
            if self._llm is not None:
                raise SchedulerError("llm_busy", "local language lane is occupied",
                                     {"state": self._llm["state"]})
            quiet_since = max(self._llm_idle_since, self._last_foreground_at or self._llm_idle_since)
            if kind == "background" and now - quiet_since < LLM_BACKGROUND_IDLE_SECONDS:
                raise SchedulerError("llm_foreground_recent", "foreground quiet time is not complete")
            try:
                lease = self._acquire_locked(owner, self._llm_resources(kind), "llm." + kind,
                                            90 if kind == "foreground" else 10, ttl_seconds)
            except SchedulerError as exc:
                if exc.code == "resource_busy":
                    raise SchedulerError("llm_busy", "local language lane is occupied") from exc
                raise
            try:
                self._write_marker(lease, kind)
            except (OSError, ValueError):
                self.leases.pop(lease.lease_id)
                self._llm_state_error = "llm_marker_unavailable"
                raise SchedulerError(self._llm_state_error, "local language lane persistence is unavailable")
            self._llm = {"token": lease.lease_id, "owner": owner, "kind": kind, "state": "active"}
            return self._llm_public_locked()

    def llm_finish(self, *, token, owner, terminal=True):
        if not isinstance(terminal, bool):
            raise SchedulerError("invalid_terminal", "terminal must be a boolean acknowledgement")
        with self.lock:
            self._purge()
            if self._llm is None or token != self._llm["token"]:
                raise SchedulerError("lease_not_found", "language lane token not found")
            if owner != self._llm["owner"]:
                raise SchedulerError("permission_denied", "language lane owner mismatch")
            if not terminal:
                self._llm["state"] = "quarantined"
                return {**self._llm_public_locked(), "released": False}
            try:
                self._remove_marker()
            except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
                self._llm["state"] = "quarantined"
                raise SchedulerError("llm_marker_unavailable", "local language lane persistence is unavailable")
            self._llm_idle_since = self.clock()
            self.leases.pop(token)
            self._llm = None
            return {"token": token, "released": True, "state": "idle"}

    def llm_status(self):
        with self.lock:
            self._purge()
            return {"state": (self._llm["state"] if self._llm is not None else
                              "quarantined" if self._llm_state_error else "idle"),
                    "lane": self._llm_public_locked() if self._llm is not None else None,
                    "persistence_error": self._llm_state_error,
                    "last_foreground_at": self._last_foreground_at,
                    "background_idle_seconds": LLM_BACKGROUND_IDLE_SECONDS,
                    "policy": "single-generation-no-implicit-preemption"}

    def _llm_public_locked(self):
        lease = self.leases[self._llm["token"]]
        return {**self._llm, "created_at": lease.created_at, "expires_at": lease.expires_at}

    def status(self):
        with self.lock:
            self._purge()
            used = self._used_locked()
            return {"capacity": dict(self.capacity), "used": used,
                    "available": {key: value - used[key] for key, value in self.capacity.items()},
                    "leases": [lease.public() for lease in self.leases.values()],
                    "policy": "admission-no-implicit-preemption"}

    def _used_locked(self):
        used = {key: 0 for key in self.capacity}
        if self._llm_state_error is not None:
            used["local_llm"] = 1
            if "npu_compute" in used:
                used["npu_compute"] = min(LLM_BACKGROUND_NPU_COMPUTE, self.capacity["npu_compute"])
        for lease in self.leases.values():
            for key, amount in lease.resources.items():
                used[key] += amount
        return used
