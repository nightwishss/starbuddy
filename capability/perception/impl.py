"""Compose camera acquisition and local QNN vision into transient observations."""

from __future__ import annotations

import secrets
import threading
import time
import json
import urllib.request
from collections import Counter

from capability.camera import impl as camera
from capability.vision import impl as vision
from perception_context_service import PerceptionContextClient


_LOCK = threading.Lock()
_PREVIOUS: dict[str, dict] = {}
_PROFILES = {
    "fast": {"transform": {"resize": {"width": 640, "height": 360}}, "quality": 72},
    "balanced": {"transform": {"resize": {"width": 960, "height": 540}}, "quality": 82},
    "detail": {"quality": 88},
}
_POSTURE_URL = "http://127.0.0.1:8792/api/posture"
_CONTEXT = PerceptionContextClient(timeout=2.0)


def _summary(detections: list[dict]) -> dict:
    counts = Counter(item["label"] for item in detections)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:20]
    primary = max(detections, key=lambda item: item["confidence"], default=None)
    return {
        "labels": [label for label, _ in ordered],
        "counts": {label: count for label, count in ordered},
        "primary_label": primary["label"] if primary else None,
        "human_present": counts.get("person", 0) > 0,
    }


def _stable_summary(counts: dict) -> dict:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:20]
    return {
        "labels": [label for label, _ in ordered],
        "counts": {label: count for label, count in ordered},
        "human_present": counts.get("person", 0) > 0,
    }


def _change(role: str, observation_id: str, summary: dict, required: int) -> tuple[dict, dict]:
    current = summary["counts"]
    with _LOCK:
        state = _PREVIOUS.get(role)
        if state is None:
            state = {
                "confirmed_id": observation_id, "confirmed": dict(current),
                "candidate": None, "candidate_hits": 0,
            }
            _PREVIOUS[role] = state
            return ({
                "changed": False, "baseline": True, "stable": True,
                "provisional": False, "added": {}, "removed": {},
                "confirmation_count": required, "confirmation_required": required,
                "previous_observation_id": None,
            }, _stable_summary(state["confirmed"]))

        confirmed = state["confirmed"]
        previous_id = state["confirmed_id"]
        if current == confirmed:
            state["candidate"] = None
            state["candidate_hits"] = 0
            return ({
                "changed": False, "baseline": False, "stable": True,
                "provisional": False, "added": {}, "removed": {},
                "confirmation_count": required, "confirmation_required": required,
                "previous_observation_id": previous_id,
            }, _stable_summary(confirmed))

        if state["candidate"] == current:
            state["candidate_hits"] += 1
        else:
            state["candidate"] = dict(current)
            state["candidate_hits"] = 1
        hits = state["candidate_hits"]
        if hits < required:
            return ({
                "changed": False, "baseline": False, "stable": False,
                "provisional": True, "added": {}, "removed": {},
                "confirmation_count": hits, "confirmation_required": required,
                "previous_observation_id": previous_id,
            }, _stable_summary(confirmed))

        labels = sorted(set(confirmed) | set(current))
        delta = {label: current.get(label, 0) - confirmed.get(label, 0) for label in labels}
        added = {label: value for label, value in delta.items() if value > 0}
        removed = {label: -value for label, value in delta.items() if value < 0}
        state.update({
            "confirmed_id": observation_id, "confirmed": dict(current),
            "candidate": None, "candidate_hits": 0,
        })
        return ({
            "changed": bool(added or removed), "baseline": False, "stable": True,
            "provisional": False, "added": added, "removed": removed,
            "confirmation_count": required, "confirmation_required": required,
            "previous_observation_id": previous_id,
        }, _stable_summary(current))


def observe(args):
    role = args.get("role", "camera.environment")
    profile = args.get("profile", "fast")
    capture_args = {"role": role, **_PROFILES[profile]}
    frame = camera.snapshot(capture_args)
    try:
        detection = vision.detect_objects({
            "media_id": frame["media_id"],
            "media_owner": camera._MEDIA_OWNER,
            "confidence": args.get("confidence", 0.35),
        })
        observation_id = f"observation_{secrets.token_urlsafe(18)}"
        summary = _summary(detection["detections"])
        change, stable_summary = _change(
            role, observation_id, summary, args.get("stability_samples", 2)
        )
        return {
            "observation_id": observation_id,
            "observed_at": time.time(),
            "role": role,
            "profile": profile,
            "frame": {
                "camera_id": frame["camera_id"],
                "width": frame["width"], "height": frame["height"],
            },
            "summary": summary,
            "stable_summary": stable_summary,
            "change": change,
            "detections": detection["detections"],
            "model": detection["model"], "backend": detection["backend"],
            "timing": detection["timing"],
        }
    finally:
        camera.release_frame({"frame_id": frame["frame_id"]})


def _posture(operation: str, args: dict) -> dict:
    if operation not in {"monitor", "stop"} and args:
        raise ValueError("posture capability does not accept arguments")
    path = f"{_POSTURE_URL}/{operation}"
    if operation == "status":
        request = urllib.request.Request(path, method="GET")
    else:
        request = urllib.request.Request(
            path, data=json.dumps(args,separators=(",", ":")).encode(), method="POST",
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(request, timeout=3.0) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError("invalid posture service response")
    return value


def context_snapshot(args):
    if args:
        raise ValueError("perception context query does not accept arguments")
    return _CONTEXT.snapshot()


ACTIONS = {
    "perception.observe": observe,
    "perception.context.snapshot": context_snapshot,
    "perception.posture.status": lambda args: _posture("status", args),
    "perception.posture.start": lambda args: _posture("start", args),
    "perception.posture.pause": lambda args: _posture("pause", args),
    "perception.posture.monitor": lambda args: _posture("monitor", args),
    "perception.posture.stop": lambda args: _posture("stop", args),
    "perception.posture.calibrate": lambda args: _posture("calibrate", args),
    "perception.posture.reset": lambda args: _posture("reset", args),
    "perception.posture.simulate": lambda args: _posture("simulate", args),
}
