"""Bounded, semantic-only fusion state. Host code owns permissions and decisions."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass


STATES = frozenset({"present", "observed_absent", "unknown", "unavailable", "disabled"})
FIELDS = {
    "gesture": {"label", "event_id", "track_id"},
    "presence": {"count", "track_ids"},
    "identity": {"label", "consent", "track_id"},
    "objects": {"counts"},
    "posture": {"state", "reasons", "enabled", "calibrated", "generation"},
    "affect": {"label", "track_id"},
    "interaction": {"phase", "turn_id", "quiet", "user_busy"},
    "output": {"busy", "state", "job_id", "queued"},
    "intent": {"snooze_until", "origin"},
    "resources": {"audio_available", "cpu_available", "npu_available"},
    "task": {"state", "task_id"},
}


@dataclass(frozen=True)
class Access:
    """Constructed by trusted host code, never accepted as a client-supplied grant."""
    session_id: str | None = None
    kinds: frozenset = frozenset(FIELDS)
    identity: bool = False


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _bounded(value, depth=0):
    if depth > 3:
        raise ValueError("semantic value too deep")
    if value is None or isinstance(value, bool):
        return value
    if _finite(value):
        return value
    if isinstance(value, str) and len(value) <= 128:
        return value
    if isinstance(value, list) and len(value) <= 20:
        return [_bounded(v, depth + 1) for v in value]
    if isinstance(value, dict) and len(value) <= 20:
        if any(not isinstance(k, str) or len(k) > 64 for k in value):
            raise ValueError("invalid semantic keys")
        forbidden = {"image", "audio", "frame", "base64", "embedding", "embeddings", "tensor", "raw"}
        if any(k.lower() in forbidden for k in value):
            raise ValueError("raw media is not semantic context")
        return {k: _bounded(v, depth + 1) for k, v in value.items()}
    raise ValueError("invalid semantic value")


class FusionCore:
    def __init__(self, *, clock=time.time, maximum=128):
        self.clock, self.maximum = clock, maximum
        self.epoch, self.revision = uuid.uuid4().hex, 0
        self.records = {}
        self.retired_epochs = {}
        self.lock = threading.RLock()
        self.counters = {"accepted": 0, "duplicate": 0, "out_of_order": 0,
                         "expired": 0, "evicted": 0, "invalidated": 0}

    def observe(self, *, provider, kind, source, observed_at, ttl_seconds,
                state="present", data=None, subject=None, session_id=None,
                quality=None, evidence_id=None, sequence=None, source_epoch=None,
                origin="poll", arrived_at=None):
        now = self.clock()
        if kind not in FIELDS or state not in STATES:
            raise ValueError("invalid observation kind/state")
        for value in (provider, source, origin):
            if not isinstance(value, str) or not 1 <= len(value) <= 128:
                raise ValueError("invalid observation source")
        for value in (subject, session_id, source_epoch, evidence_id):
            if value is not None and (not isinstance(value, str) or not 1 <= len(value) <= 192):
                raise ValueError("invalid observation reference")
        if (not _finite(observed_at) or observed_at > now + 1
                or not _finite(ttl_seconds) or not .1 <= ttl_seconds <= 3600):
            raise ValueError("invalid observation time")
        if quality is not None and (not _finite(quality) or not 0 <= quality <= 1):
            raise ValueError("invalid observation quality")
        if sequence is not None and (not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0):
            raise ValueError("invalid observation sequence")
        data = {} if data is None else data
        if not isinstance(data, dict) or set(data) - FIELDS[kind]:
            raise ValueError("unsupported semantic fields")
        data = _bounded(data)
        def short_text(value):
            return isinstance(value, str) and 0 < len(value) <= 128
        def text_list(value):
            return isinstance(value, list) and all(short_text(v) for v in value)
        # Field types, not just key names, make this a semantic contract. Do
        # not allow an embedding/tensor to masquerade as a label or count.
        for field in {"label", "consent", "track_id", "origin", "state", "task_id", "job_id", "phase", "turn_id"} & data.keys():
            if data[field] is not None and not short_text(data[field]):
                raise ValueError("invalid semantic text")
        for field in {"track_ids", "reasons"} & data.keys():
            if not text_list(data[field]):
                raise ValueError("invalid semantic text list")
        if "counts" in data and (not isinstance(data["counts"], dict) or any(
                not short_text(k) or type(v) is not int or not 0 <= v <= 1000
                for k, v in data["counts"].items())):
            raise ValueError("invalid object counts")
        if "generation" in data and (type(data["generation"]) is not int or data["generation"] < 0):
            raise ValueError("invalid posture generation")
        if "queued" in data and (type(data["queued"]) is not int or not 0 <= data["queued"] <= 1_000_000):
            raise ValueError("invalid output queue count")
        if state in {"present", "observed_absent"}:
            if kind == "presence" and (type(data.get("count")) is not int or not 0 <= data["count"] <= 20
                    or not isinstance(data.get("track_ids", []), list)):
                raise ValueError("invalid presence evidence")
            if kind == "posture" and (data.get("state") not in {"healthy", "suspect", "unhealthy", "unknown", "observing", "calibrating"}
                    or type(data.get("enabled")) is not bool or type(data.get("calibrated")) is not bool):
                raise ValueError("invalid posture evidence")
            if kind == "interaction" and data.get("phase") not in {"idle", "listening", "thinking", "speaking", "acting", "attention", "error"}:
                raise ValueError("invalid interaction evidence")
            if kind == "interaction" and any(type(data[k]) is not bool for k in
                                             ("quiet", "user_busy") if k in data):
                raise ValueError("invalid interaction flags")
            if kind == "output" and type(data.get("busy")) is not bool:
                raise ValueError("invalid output evidence")
            if kind == "resources" and (type(data.get("audio_available")) is not bool
                    or not _finite(data.get("cpu_available")) or data["cpu_available"] < 0
                    or ("npu_available" in data and (not _finite(data["npu_available"])
                        or not 0 <= data["npu_available"] <= 100))):
                raise ValueError("invalid resource evidence")
            if kind == "intent" and (not _finite(data.get("snooze_until")) or data["snooze_until"] < observed_at):
                raise ValueError("invalid intent evidence")
        if kind == "identity" and state == "present" and data.get("consent") != "enrolled_local":
            raise ValueError("identity requires consent")
        if kind == "gesture" and (data.get("label") not in {"wave", "arrival"}
                or not isinstance(data.get("event_id"), str) or len(data["event_id"]) != 32
                or ttl_seconds > 3):
            raise ValueError("invalid short-lived gesture")
        if kind == "intent" and data.get("origin") != "user":
            raise ValueError("robot/model output is not user intent")
        if state == "observed_absent" and (kind != "presence" or data.get("count") != 0):
            raise ValueError("absence needs an explicit empty presence observation")
        arrived_at = now if arrived_at is None else arrived_at
        if not _finite(arrived_at) or arrived_at > now + 1:
            raise ValueError("invalid arrival time")
        record = dict(provider=provider, kind=kind, source=source, subject=subject,
                      session_id=session_id, observed_at=float(observed_at),
                      arrived_at=float(arrived_at), expires_at=observed_at + ttl_seconds,
                      quality=quality, state=state, data=data, sequence=sequence,
                      source_epoch=source_epoch, origin=origin)
        identity = {k: v for k, v in record.items() if k != "arrived_at"}
        record["evidence_id"] = evidence_id or hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
        key = (provider, kind, source, subject, session_id)
        with self.lock:
            previous = self.records.get(key)
            if source_epoch in self.retired_epochs.get(key, ()):
                self.counters["out_of_order"] += 1
                return False
            if previous and previous["evidence_id"] == record["evidence_id"]:
                self.counters["duplicate"] += 1
                return False
            if previous and previous["source_epoch"] == source_epoch:
                older_sequence = (sequence is not None and previous["sequence"] is not None
                                  and sequence <= previous["sequence"])
                if observed_at < previous["observed_at"] or older_sequence:
                    self.counters["out_of_order"] += 1
                    return False
            if previous and previous["source_epoch"] != source_epoch:
                retired = self.retired_epochs.setdefault(key, [])
                retired.append(previous["source_epoch"])
                del retired[:-4]
            self.records[key] = record
            if len(self.records) > self.maximum:
                oldest = min(self.records, key=lambda k: self.records[k]["arrived_at"])
                del self.records[oldest]
                self.retired_epochs.pop(oldest, None)
                self.counters["evicted"] += 1
            self.counters["accepted"] += 1
            self.revision += 1
            self.tick(now)
        return True

    def tick(self, now=None):
        """Called by the existing host loop even when no events are arriving."""
        now = self.clock() if now is None else now
        with self.lock:
            changed = 0
            for record in self.records.values():
                if record["expires_at"] <= now and record["state"] != "stale":
                    record.update(state="stale", data={})
                    changed += 1
            if changed:
                self.revision += 1
                self.counters["expired"] += changed
            return changed

    def invalidate(self, *, origin="event", reason="bus_gap"):
        with self.lock:
            for record in self.records.values():
                if record["origin"] == origin:
                    record.update(state="unknown", data={}, invalidated_by=reason,
                                  expires_at=self.clock() + 3)
            self.revision += 1
            self.counters["invalidated"] += 1

    @staticmethod
    def _visible(record, access):
        return (record["kind"] in access.kinds
                and (record["session_id"] is None or record["session_id"] == access.session_id)
                and (record["kind"] != "identity" or access.identity))

    def snapshot(self, access=Access()):
        with self.lock:
            self.tick()
            records = [deepcopy(r) for r in self.records.values() if self._visible(r, access)]
            version = f"{self.epoch}:{self.revision}"
        groups = {}
        for r in records:
            groups.setdefault((r["kind"], r["source"], r["subject"], r["session_id"]), []).append(r)
        items = []
        for key, evidence in groups.items():
            active = [r for r in evidence if r["state"] not in {"stale", "unknown", "unavailable", "disabled"}]
            values = {json.dumps([r["state"], r["data"]], sort_keys=True) for r in active}
            chosen = max(evidence, key=lambda r: r["arrived_at"])
            conflict = len(values) > 1
            item = {k: chosen[k] for k in ("kind", "source", "subject", "session_id",
                                           "state", "observed_at", "arrived_at", "expires_at", "data")}
            if conflict:
                item.update(state="unknown", data={})
            item.update(conflict=conflict, evidence=[r["evidence_id"] for r in evidence],
                        qualities=[r["quality"] for r in evidence],
                        providers=[r["provider"] for r in evidence])
            items.append(item)
        return {"version": version, "epoch": self.epoch, "generated_at": self.clock(),
                "items": items, "semantics": "observations_not_ground_truth"}

    def explain(self, evidence_id, access=Access()):
        with self.lock:
            self.tick()
            for record in self.records.values():
                if record["evidence_id"] == evidence_id and self._visible(record, access):
                    return deepcopy(record)
        return None

    def stats(self):
        with self.lock:
            return {"ready": True, "epoch": self.epoch, "revision": self.revision,
                    "records": len(self.records), "maximum": self.maximum, **self.counters}


class EventCursor:
    """Bounded-page cursor, including bus boot identity and retention loss."""
    def __init__(self, core):
        self.core, self.epoch, self.revision = core, None, 0
        self.gaps, self.restarts = 0, 0

    def consume(self, page, accept):
        epoch = page.get("epoch")
        reset = (self.epoch is not None and epoch != self.epoch) or page.get("reset", False)
        if reset:
            self.core.invalidate(reason="bus_restart")
            self.restarts += 1
            self.revision, self.epoch = 0, epoch
            # Refetch from zero: this page may have been requested with an old
            # cursor already overtaken by a new bus, so its prefix is missing.
            return True
        if page.get("gap"):
            self.core.invalidate(reason="bus_gap")
            self.gaps += 1
        self.epoch = epoch
        for event in page.get("events", []):
            if event.get("revision", 0) > self.revision:
                accept(event, epoch)
        self.revision = page.get("next_revision", page.get("revision", self.revision))
        return bool(page.get("has_more"))
