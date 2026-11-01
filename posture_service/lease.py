"""Persistent single-owner lease for the monitor_posture Skill."""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from copy import deepcopy
from pathlib import Path


class PostureLeaseError(ValueError):
    pass


class PostureMonitorLease:
    def __init__(self, posture, state_path: Path, *, events=None, clock=time.time,
                 autostart=True, start_worker=True):
        self.posture = posture
        self.state_path = Path(state_path)
        self.events = events
        self.clock = clock
        self.condition = threading.Condition(threading.RLock())
        self.closed = False
        self.start_error = None
        self.state = self._load("manual" if autostart else "paused")
        self._restore()
        self.thread = None
        if start_worker:
            self.thread = threading.Thread(
                target=self._worker, name="vq-posture-lease", daemon=True,
            )
            self.thread.start()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=2)

    def status(self) -> dict:
        value = self.posture.status()
        with self.condition:
            state = deepcopy(self.state)
        session = None
        if state["mode"] == "skill":
            session = {key: state[key] for key in (
                "lease_id", "task_id", "started_at", "expires_at",
                "duration_seconds", "bad_duration_seconds", "cooldown_seconds",
            )}
            session["remaining_seconds"] = max(0, round(state["expires_at"] - self.clock()))
        value["monitor_session"] = session
        value["monitor_mode"] = state["mode"]
        value["monitor_start_error"] = self.start_error
        return value

    def _start_controller(self):
        try:
            result = self.posture.start()
        except Exception:
            # Keep the user's persisted intent and known baseline. The HTTP
            # service must remain available so an explicit retry can recover
            # a camera/pose dependency which was not ready at boot.
            self.start_error = "posture_start_failed"
            raise
        self.start_error = None
        return result

    def monitor(self, *, task_id, duration_seconds, bad_duration_seconds=20,
                cooldown_seconds=180) -> dict:
        values = (duration_seconds, bad_duration_seconds, cooldown_seconds)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise PostureLeaseError("monitor parameters must be integers")
        if not 60 <= duration_seconds <= 14_400:
            raise PostureLeaseError("duration_seconds out of range")
        if not 5 <= bad_duration_seconds <= 120:
            raise PostureLeaseError("bad_duration_seconds out of range")
        if not 30 <= cooldown_seconds <= 1_800:
            raise PostureLeaseError("cooldown_seconds out of range")
        if not isinstance(task_id, str) or not re.fullmatch(r"task_[0-9a-f]{32}", task_id):
            raise PostureLeaseError("invalid task_id")
        if not self.posture.status().get("calibrated"):
            raise PostureLeaseError("posture calibration required")
        now = self.clock()
        with self.condition:
            if self.state["mode"] == "skill" and self.state["expires_at"] > now:
                if self.state["task_id"] == task_id:
                    return {**self.status(), "accepted": True, "idempotent": True}
                raise PostureLeaseError("another posture monitor is active")
            self.posture.configure_monitor_policy(
                confirmation_seconds=bad_duration_seconds,
                cooldown_seconds=cooldown_seconds,
            )
            self._start_controller()
            self.state = {
                "version": 1, "mode": "skill", "lease_id": "posture_" + secrets.token_hex(12),
                "task_id": task_id, "started_at": now, "expires_at": now + duration_seconds,
                "duration_seconds": duration_seconds,
                "bad_duration_seconds": bad_duration_seconds,
                "cooldown_seconds": cooldown_seconds,
            }
            self._write()
            self.condition.notify_all()
            result = self.status()
        self._publish("skill.monitor_posture.started", self.state)
        return {**result, "accepted": True, "idempotent": False}

    def manual_start(self) -> dict:
        with self.condition:
            # Keep the desired persisted mode and the controller transition in
            # one serialized operation. An expiring skill must not pause a
            # newer manual enable after it has been acknowledged to the UI.
            self._start_controller()
            self.state = {"version": 1, "mode": "manual"}
            self._write()
            self.condition.notify_all()
            return self.status()

    def manual_pause(self) -> dict:
        with self.condition:
            self.posture.pause()
            self.start_error = None
            self.state = {"version": 1, "mode": "paused"}
            self._write()
            self.condition.notify_all()
            return self.status()

    def stop(self, *, task_id: str) -> dict:
        """Stop only the lease owned by this task; never pause another owner."""
        if not isinstance(task_id, str) or not re.fullmatch(r"task_[0-9a-f]{32}", task_id):
            raise PostureLeaseError("invalid task_id")
        with self.condition:
            if self.state["mode"] != "skill":
                return {**self.status(), "stopped": False}
            if self.state["task_id"] != task_id:
                raise PostureLeaseError("another posture monitor is active")
            stopped = deepcopy(self.state)
            self.state = {"version": 1, "mode": "paused"}
            self._write()
            self.condition.notify_all()
            result = self.posture.pause()
            self.start_error = None
        self._publish("skill.monitor_posture.stopped", stopped)
        return {**result, "monitor_mode": "paused", "monitor_session": None, "stopped": True}

    def expire_due(self) -> bool:
        now = self.clock()
        with self.condition:
            if self.state["mode"] != "skill" or self.state["expires_at"] > now:
                return False
            completed = deepcopy(self.state)
            self.posture.pause()
            self.start_error = None
            self.state = {"version": 1, "mode": "paused"}
            self._write()
        self._publish("skill.monitor_posture.completed", completed)
        return True

    def _restore(self):
        try:
            self._restore_controller()
        except Exception:
            # This is a one-time restore attempt, not an automatic retry loop.
            # Actual enabled/source state still comes from the controller.
            self.start_error = "posture_start_failed"

    def _restore_controller(self):
        if self.state["mode"] == "skill":
            if self.state["expires_at"] <= self.clock():
                self.state = {"version": 1, "mode": "paused"}
                self._write()
                self.posture.pause()
                return
            self.posture.configure_monitor_policy(
                confirmation_seconds=self.state["bad_duration_seconds"],
                cooldown_seconds=self.state["cooldown_seconds"],
            )
            self._start_controller()
        elif self.state["mode"] == "manual":
            self._start_controller()
        else:
            self.posture.pause()

    def _worker(self):
        while True:
            with self.condition:
                if self.closed:
                    return
                wait = 30.0
                if self.state["mode"] == "skill":
                    wait = max(0.05, min(30.0, self.state["expires_at"] - self.clock()))
                self.condition.wait(wait)
            self.expire_due()

    def _load(self, default_mode):
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": 1, "mode": default_mode}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise PostureLeaseError("invalid posture lease state") from exc
        if not isinstance(value, dict) or value.get("version") != 1 or value.get("mode") not in {
            "manual", "paused", "skill",
        }:
            raise PostureLeaseError("invalid posture lease state")
        if value["mode"] == "skill":
            required = {"lease_id", "task_id", "started_at", "expires_at", "duration_seconds",
                        "bad_duration_seconds", "cooldown_seconds"}
            if not required <= value.keys():
                raise PostureLeaseError("invalid posture lease state")
        return value

    def _write(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, separators=(",", ":")), encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.state_path)

    def _publish(self, topic, state):
        if self.events is None:
            return
        payload = {key: state[key] for key in (
            "lease_id", "task_id", "started_at", "expires_at", "duration_seconds",
        )}
        try:
            self.events.publish(topic, "posture-skill", payload,
                                privacy="local-private", trace_id=state["task_id"])
        except Exception:
            pass
