"""Pure posture proposals: no model calls, scheduling, audio or hardware access."""

from __future__ import annotations

import hashlib
import json


def posture_proposal(snapshot, *, now, session_id=None):
    items = snapshot["items"]
    def current(kind, source=None):
        selected = [v for v in items if v["kind"] == kind and
                    (source is None or v["source"] == source)]
        return max(selected, key=lambda v: v["arrived_at"]) if selected else None
    posture = current("posture", "camera.environment")
    interaction, output = current("interaction"), current("output", "output-arbiter")
    resources, intent = current("resources"), current("intent")
    presence = current("presence", "camera.environment")
    refs = [ref for item in (posture, interaction, output, resources, intent, presence)
            if item for ref in item["evidence"]]
    decision = {"action": "suppress", "reason": "posture_unknown",
                "context_version": snapshot["version"], "session_id": session_id,
                "subject": "camera.environment:unbound", "evidence": refs,
                "expires_at": now + 2, "observation_is_identity": False}
    def finish(action, reason):
        decision.update(action=action, reason=reason)
        decision["decision_id"] = hashlib.sha256(json.dumps(decision, sort_keys=True).encode()).hexdigest()[:24]
        return decision
    if posture and posture["expires_at"] <= now:
        return finish("suppress", "posture_stale")
    if not posture or posture["state"] != "present":
        return finish("suppress", "posture_" + (posture["state"] if posture else "unknown"))
    data = posture["data"]
    if not data.get("enabled"):
        return finish("suppress", "posture_disabled")
    if not data.get("calibrated") or data.get("state") != "unhealthy":
        return finish("suppress", "no_confirmed_posture_reminder")
    # No cross-camera identity association. A generic posture reminder is
    # source-scoped; multiple subjects in that source make it ambiguous.
    # This producer detects FACES. An explicit no-face observation is not
    # evidence that no torso/person is present (the posture camera can crop
    # the head). The independent fresh Pose evidence remains authoritative
    # for a generic, source-scoped posture proposal, never a named person.
    if presence and presence["expires_at"] > now and presence["state"] == "present" and presence["data"].get("count", 0) > 1:
        return finish("suppress", "ambiguous_subject")
    if not intent or intent["state"] != "present" or intent["expires_at"] <= now:
        return finish("defer", "intent_unknown")
    if intent and intent["expires_at"] > now and intent["state"] == "present" and intent["data"].get("snooze_until", 0) > now:
        return finish("defer", "user_snoozed")
    if not interaction or interaction["state"] != "present" or interaction["expires_at"] <= now:
        return finish("defer", "interaction_unknown")
    if interaction["session_id"] != session_id:
        return finish("defer", "session_changed")
    if interaction["data"].get("phase") != "idle":
        return finish("defer", "conversation_busy")
    if not output or output["state"] != "present" or output["expires_at"] <= now:
        return finish("defer", "output_unknown")
    if output["data"].get("busy"):
        return finish("defer", "robot_speaking")
    if not resources or resources["state"] != "present" or resources["expires_at"] <= now:
        return finish("defer", "resources_unknown")
    if not resources["data"].get("audio_available") or resources["data"].get("cpu_available", 0) <= 0:
        return finish("defer", "resource_busy")
    if resources["data"].get("npu_available", 100) < 100:
        return finish("defer", "compute_busy")
    decision["expires_at"] = min(v["expires_at"] for v in (posture, interaction, output, resources, intent))
    decision["posture_generation"] = data.get("generation")
    return finish("propose", "posture_reminder_ready")
