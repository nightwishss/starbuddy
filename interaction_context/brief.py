"""Ephemeral anonymous semantic changes; never a source of current facts."""

from __future__ import annotations

import math
import re
import threading
import time
from copy import deepcopy


SOURCES = frozenset({"camera.front", "camera.environment"})
KINDS = frozenset({"presence", "posture", "objects"})
REASONS = frozenset({"head_forward", "torso_lean", "shoulder_asymmetry", "too_close", "too_far"})
OBJECTS = frozenset("person|bicycle|car|motorcycle|airplane|bus|train|truck|boat|traffic light|"
                   "fire hydrant|stop sign|parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|"
                   "bear|zebra|giraffe|backpack|umbrella|handbag|tie|suitcase|frisbee|skis|snowboard|"
                   "sports ball|kite|baseball bat|baseball glove|skateboard|surfboard|tennis racket|"
                   "bottle|wine glass|cup|fork|knife|spoon|bowl|banana|apple|sandwich|orange|broccoli|"
                   "carrot|hot dog|pizza|donut|cake|chair|couch|potted plant|bed|dining table|toilet|"
                   "tv|laptop|mouse|remote|keyboard|cell phone|microwave|oven|toaster|sink|refrigerator|"
                   "book|clock|vase|scissors|teddy bear|hair drier|toothbrush".split("|"))


def _finite(value):
    try:
        return type(value) in {int, float} and math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _payload(kind, state, data):
    """Return only supported enums/numbers, plus a private continuity key."""
    if not isinstance(data, dict):
        return None
    if kind == "presence":
        count, tracks = data.get("count"), data.get("track_ids", [])
        if (type(count) is not int or count not in {0, 1} or not isinstance(tracks, list)
                or len(tracks) != count or any(not isinstance(t, str) or not re.fullmatch(
                    r"[A-Za-z0-9_.:-]{1,96}", t) for t in tracks)
                or (count == 0) != (state == "observed_absent")):
            return None
        return {"count": count}, tuple(sorted(tracks))
    if state != "present":
        return None
    if kind == "posture":
        generation = data.get("generation")
        if (not isinstance(data.get("state"), str)
                or data["state"] not in {"healthy", "suspect", "unhealthy", "observing", "calibrating"}
                or data.get("enabled") is not True or type(data.get("calibrated")) is not bool
                or type(generation) is not int or not 0 <= generation < 2 ** 63):
            return None
        reasons = data.get("reasons", [])
        if not isinstance(reasons, list):
            return None
        return {"state": data["state"], "enabled": True, "calibrated": data["calibrated"],
                "reasons": sorted({r for r in reasons[:20] if isinstance(r, str) and r in REASONS})}, generation
    counts = data.get("counts")
    if not isinstance(counts, dict) or len(counts) > 20:
        return None
    original_counts = counts
    counts = {k: v for k, v in counts.items()
              if isinstance(k, str) and k in OBJECTS and type(v) is int and 0 <= v <= 1000}
    if (original_counts and not counts) or counts.get("person", 0) > 1:
        return None
    return {"counts": dict(sorted(counts.items()))}, None


def _project(snapshot, now):
    """Invalid/ambiguous evidence invalidates its entire camera source."""
    result, invalid = {}, set()
    if not isinstance(snapshot, dict):
        return {}
    items = snapshot.get("items", [])
    if not isinstance(items, list):
        return {}
    for item in items[:128]:
        if (not isinstance(item, dict) or not isinstance(item.get("source"), str)
                or item["source"] not in SOURCES or not isinstance(item.get("kind"), str)
                or item["kind"] not in KINDS or item.get("session_id") is not None
                or item.get("subject") is not None):
            continue
        source, kind = item["source"], item["kind"]
        target = result.setdefault(source, {"items": {}, "tracks": None, "generation": None})
        observed, expires = item.get("observed_at"), item.get("expires_at")
        normalized = _payload(kind, item.get("state"), item.get("data"))
        if (not isinstance(item.get("state"), str)
                or item["state"] not in {"present", "observed_absent"} or item.get("conflict")
                or not _finite(observed) or not _finite(expires) or observed > now + 1
                or not observed < expires or expires <= now or normalized is None
                or kind in target["items"]):
            invalid.add(source)
            continue
        payload, continuity = normalized
        target["items"][kind] = {"data": payload, "observed_at": observed, "expires_at": expires}
        if kind == "presence":
            target["tracks"] = continuity
        elif kind == "posture":
            target["generation"] = continuity
    for source in invalid:
        result.pop(source, None)
    for target in result.values():
        target["key"] = (target.pop("tracks"), target.pop("generation"), tuple(sorted(target["items"])))
        target["fingerprint"] = {k: v["data"] for k, v in target["items"].items()}
    return result


def sanitize_change(change):
    """Public projection only; callers still enforce epoch, TTL and permissions.

    Continuity was checked by SemanticBrief.read; track IDs never cross this
    contract, so this stateless helper cannot grant subject association.
    """
    if (not isinstance(change, dict) or not isinstance(change.get("source"), str)
            or change["source"] not in SOURCES or not isinstance(change.get("kind"), str)
            or change["kind"] not in KINDS):
        return None
    observed, expires, retain = (change.get(k) for k in
                                  ("observed_at", "source_expires_at", "retain_until"))
    if (not all(_finite(v) for v in (observed, expires, retain))
            or not observed < expires <= observed + 3600 or not observed < retain <= observed + 300):
        return None
    result = {k: change[k] for k in ("source", "kind", "observed_at", "source_expires_at", "retain_until")}
    for field in ("before", "after"):
        data = change.get(field)
        if not isinstance(data, dict):
            return None
        if change["kind"] == "presence":
            count = data.get("count")
            if type(count) is not int or count not in {0, 1}:
                return None
            result[field] = {"count": count}
        else:
            payload = ({k: data.get(k) for k in ("state", "reasons", "enabled", "calibrated")}
                       | {"generation": 0}) if change["kind"] == "posture" else data
            normalized = _payload(change["kind"], "present", payload)
            if normalized is None:
                return None
            result[field] = normalized[0]
    return result if result["before"] != result["after"] else None


class SemanticBrief:
    """An existing host tick prepares; foreground reads only filter/copy."""

    def __init__(self, clock=time.time, interval=30, retention=300, maximum=64, enabled=True):
        if (not _finite(interval) or interval <= 0 or not _finite(retention) or retention <= 0
                or type(maximum) is not int or maximum < 1):
            raise ValueError("invalid semantic brief bounds")
        self.clock, self.interval = clock, interval
        self.retention, self.maximum = min(retention, 300), min(maximum, 64)
        self.enabled, self.closed = bool(enabled), False
        self.lock = threading.RLock()
        self._boundary, self._epoch, self._prepared_at = None, None, None
        self._previous, self._history, self._published, self._basis = {}, [], [], {}
        self._counts = dict(checks=0, preparations=0, skipped_unchanged=0,
                            changes=0, invalidations=0, out_of_order=0, evicted=0)

    def _clear(self):
        self._previous.clear()
        self._history.clear()
        self._published.clear()
        self._basis.clear()
        self._prepared_at = None

    def _drop_source(self, source):
        self._previous.pop(source, None)
        self._basis.pop(source, None)
        self._history[:] = [r for r in self._history if r["source"] != source]
        self._published[:] = [r for r in self._published if r["source"] != source]

    def invalidate(self):
        with self.lock:
            self._clear()
            self._boundary, self._epoch = None, None
            self._counts["invalidations"] += 1

    def close(self):
        with self.lock:
            self.closed = True
            self.invalidate()

    def update(self, snapshot, boundary):
        """Return whether a new brief was prepared, without I/O or model calls."""
        now = self.clock()
        with self.lock:
            self._counts["checks"] += 1
            if not self.enabled or self.closed:
                self._clear()
                return False
            epoch = snapshot.get("epoch") if isinstance(snapshot, dict) else None
            if not isinstance(epoch, str) or not epoch or len(epoch) > 128:
                self.invalidate()
                return False
            if boundary != self._boundary or epoch != self._epoch:
                self._clear()
                self._boundary, self._epoch = deepcopy(boundary), epoch
                self._counts["invalidations"] += 1
            current = _project(snapshot, now)
            for source in set(self._previous) - set(current):
                self._drop_source(source)
            self._history[:] = [r for r in self._history if r["retain_until"] > now]
            for source, projected in current.items():
                previous = self._previous.get(source)
                if previous and any(k in previous["items"] and (
                        v["observed_at"] < previous["items"][k]["observed_at"] or
                        (v["observed_at"] == previous["items"][k]["observed_at"]
                         and v["data"] != previous["items"][k]["data"]))
                        for k, v in projected["items"].items()):
                    self._counts["out_of_order"] += 1
                    continue
                if previous and (previous["key"] != projected["key"] or
                        any(v["expires_at"] <= now for v in previous["items"].values())):
                    self._drop_source(source)
                    previous = None
                for kind, value in projected["items"].items():
                    before = previous["items"].get(kind) if previous else None
                    if before and before["data"] != value["data"]:
                        self._history.append(dict(source=source, kind=kind,
                            before=deepcopy(before["data"]), after=deepcopy(value["data"]),
                            observed_at=value["observed_at"], source_expires_at=value["expires_at"],
                            retain_until=value["observed_at"] + self.retention))
                        self._counts["changes"] += 1
                    elif before:
                        self._counts["skipped_unchanged"] += 1
                self._previous[source] = projected
            self._history[:] = [r for r in self._history if r["retain_until"] > now]
            overflow = max(0, len(self._history) - self.maximum)
            self._counts["evicted"] += overflow
            if overflow:
                del self._history[:overflow]
            if self._prepared_at is not None and now - self._prepared_at < self.interval:
                return False
            basis = {s: {"key": p["key"], "fingerprint": p["fingerprint"]}
                     for s, p in self._previous.items()}
            if (self._prepared_at is not None and self._history == self._published
                    and basis == self._basis):
                self._counts["skipped_unchanged"] += 1
                return False
            self._published = deepcopy(self._history)
            self._basis = deepcopy(basis)
            self._prepared_at = now
            self._counts["preparations"] += 1
            return True

    def read(self, snapshot, boundary):
        now = self.clock()
        with self.lock:
            changes = []
            permitted = (self.enabled and not self.closed and isinstance(snapshot, dict)
                         and boundary == self._boundary
                         and snapshot.get("epoch") == self._epoch)
            if permitted:
                current = _project(snapshot, now)
                allowed = {s for s, p in current.items() if s in self._basis and s in self._previous
                           and self._basis[s]["key"] == p["key"]
                           and self._basis[s]["fingerprint"] == p["fingerprint"]
                           and self._previous[s]["key"] == p["key"]
                           and self._previous[s]["fingerprint"] == p["fingerprint"]
                           and all(v["expires_at"] > now for v in self._previous[s]["items"].values())}
                changes = [r for r in self._published if r["source"] in allowed and r["retain_until"] > now]
            return dict(schema_version=1, kind="historical_observations",
                        prepared_at=self._prepared_at if permitted else None,
                        basis_epoch=self._epoch if permitted else None,
                        changes=deepcopy(changes[-6:]), stats=self.stats())

    def stats(self):
        with self.lock:
            return {**self._counts, "records": len(self._history), "maximum": self.maximum,
                    "enabled": self.enabled and not self.closed}
