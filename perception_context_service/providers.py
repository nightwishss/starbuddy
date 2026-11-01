"""Adapters from existing semantic status APIs into the context contract."""

from __future__ import annotations

import json
import threading
import time
import urllib.request


DEFAULT_ENDPOINTS = {
    "face": "http://127.0.0.1:8794/status",
    "identity": "http://127.0.0.1:8795/status",
    "posture": "http://127.0.0.1:8792/api/posture/status",
}


def face_observations(value, now):
    result = []
    for source in value.get("sources", []):
        if not isinstance(source, dict) or not source.get("ready"):
            continue
        observed_at = source.get("last_frame_at")
        if not isinstance(observed_at, (int, float)):
            continue
        tracks = [{
            "track_id": item.get("track_id"), "state": item.get("state"),
            "confidence": item.get("score"),
        } for item in source.get("tracks", []) if isinstance(item, dict)]
        confidence = max((item["confidence"] for item in tracks), default=None)
        result.append({
            "provider": "face-yunet", "kind": "face_presence", "source": source.get("source"),
            "observed_at": observed_at,
            "ttl_seconds": 3.0, "confidence": confidence, "inference": True,
            "data": {"tracks": tracks},
        })
    return result


def identity_observations(value, now):
    if not value.get("ready") or not isinstance(value.get("observation"), dict):
        return []
    observation = value["observation"]
    status = observation.get("status")
    if status not in {"known", "unknown", "no_face", "confirming", "low_quality"}:
        return []
    age = value.get("last_frame_seconds_ago")
    if not isinstance(age, (int, float)) or age < 0:
        return []
    observed_at = now - age
    data = {"status": status}
    if status == "known":
        data.update(label=observation.get("label"), consent="enrolled_local")
    confidence = observation.get("similarity") if status == "known" else None
    return [{
        "provider": "face-sface", "kind": "identity", "source": value.get("source"),
        "observed_at": observed_at, "ttl_seconds": 3.0, "confidence": confidence,
        "inference": True, "data": data,
    }]


def posture_observations(value, now):
    if not isinstance(value, dict) or not isinstance(value.get("state"), str):
        return []
    pose = value.get("pose") if isinstance(value.get("pose"), dict) else {}
    features = pose.get("posture_features") if isinstance(pose.get("posture_features"), dict) else {}
    confidence = features.get("confidence")
    observed_at = value.get("observed_at")
    if not isinstance(observed_at, (int, float)):
        return []
    return [{
        "provider": "posture-policy", "kind": "posture",
        "source": ((value.get("camera") or {}).get("role")
                   if isinstance(value.get("camera"), dict) else None) or "camera.environment",
        "observed_at": observed_at, "ttl_seconds": 3.0, "confidence": confidence,
        "inference": True, "data": {
            "state": value["state"], "reasons": value.get("reasons") or [],
            "enabled": value.get("enabled", False), "calibrated": value.get("calibrated", False),
            "generation": value.get("generation"), "source_state": value.get("source_state", "unknown"),
            "frame_sequence": value.get("frame_sequence"),
        },
    }]


NORMALIZERS = {
    "face": face_observations,
    "identity": identity_observations,
    "posture": posture_observations,
}


class ProviderPoller:
    def __init__(self, context, *, endpoints=None, interval=1.0, timeout=0.75,
                 clock=time.time, opener=urllib.request.urlopen, fusion=None):
        self.context = context
        self.endpoints = endpoints or DEFAULT_ENDPOINTS
        self.interval = interval
        self.timeout = timeout
        self.clock = clock
        self.opener = opener
        self.fusion = fusion
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.providers = {name: {
            "ready": False, "last_success_at": None, "last_error": None, "failures": 0,
        } for name in self.endpoints}
        self.thread = threading.Thread(target=self._run, name="vq-perception-providers", daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3)

    def status(self):
        with self.lock:
            return {name: dict(value) for name, value in self.providers.items()}

    def poll_once(self):
        now = self.clock()
        for name, endpoint in self.endpoints.items():
            try:
                with self.opener(endpoint, timeout=self.timeout) as response:
                    value = json.load(response)
                for observation in NORMALIZERS[name](value, now):
                    self.context.observe(**observation)
                with self.lock:
                    self.providers[name].update(
                        ready=bool(value.get("ready", True)),
                        enabled=value.get("enabled", True),
                        source_state=value.get("source_state"),
                        sources=[{"source": s.get("source"), "ready": bool(s.get("ready"))}
                                 for s in value.get("sources", []) if isinstance(s, dict)][:4],
                        last_success_at=now, last_error=None, failures=0,
                    )
            except Exception as exc:
                with self.lock:
                    state = self.providers[name]
                    state.update(ready=False, last_error=type(exc).__name__,
                                 failures=state["failures"] + 1)
        if self.fusion is not None:
            snapshot = self.context.snapshot()
            snapshot["providers"] = self.status()
            try:
                self.fusion.tick(snapshot)
            except Exception:
                # Keep the existing provider loop alive; cached observations
                # still expire even when an optional fusion adapter fails.
                self.fusion.core.tick()

    def _run(self):
        while not self.stop_event.is_set():
            started = time.monotonic()
            self.poll_once()
            self.stop_event.wait(max(0.0, self.interval - (time.monotonic() - started)))
