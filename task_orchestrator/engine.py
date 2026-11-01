"""Task lifecycle, scheduler and sequential Runtime dispatch."""

from __future__ import annotations

import json
import hashlib
import logging
import threading
import time
import uuid
import re
from copy import deepcopy
from contextlib import nullcontext
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .planner import PlanError
from .client import request_task_id
from .store import TERMINAL_STATES, TaskStore, TaskStoreError


LOG = logging.getLogger("vq.task-orchestrator")


class TaskOrchestratorError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class TaskOrchestrator:
    def __init__(self, planner, store: TaskStore, runtime, events, *, skills=None, clock=time.time):
        self.planner = planner
        self.store = store
        self.runtime = runtime
        self.events = events
        self.skills = skills
        self.clock = clock
        self.condition = threading.Condition()
        self.execution_lock = threading.Lock()
        self.cancel_lock = threading.Lock()
        self.create_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._scheduler_loop, name="vq-task-scheduler", daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=3)

    def create(self, utterance: str, *, timezone="UTC", source_conversation_id=None,
               proposal=None, request_id=None) -> dict:
        """Create once per session/request key while its task record is retained.

        The existing bounded store also bounds duplicate suppression: a pruned
        terminal task no longer reserves its request key. Idempotent callers
        must provide an explicit proposal, so model inference never holds the
        creation lock and retries do not regenerate plans or relative schedules.
        """
        if source_conversation_id is not None and (
            not isinstance(source_conversation_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}", source_conversation_id)
        ):
            raise TaskOrchestratorError("invalid_request", "invalid source conversation id")
        try:
            ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, TypeError, ValueError) as exc:
            raise TaskOrchestratorError("invalid_request", "invalid timezone") from exc
        task_id = fingerprint = None
        if request_id is not None:
            if (not isinstance(request_id, str)
                    or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}", request_id)):
                raise TaskOrchestratorError("invalid_request", "invalid create request id")
            if not isinstance(proposal, dict):
                raise TaskOrchestratorError("invalid_request", "create request id requires an explicit proposal")
            if not isinstance(utterance, str) or not utterance.strip() or len(utterance) > 1000:
                raise TaskOrchestratorError("invalid_request", "utterance must be 1..1000 characters")
            proposal = deepcopy(proposal)
            try:
                canonical = json.dumps({
                    "utterance": utterance, "timezone": timezone, "proposal": proposal,
                    "source_conversation_id": source_conversation_id,
                }, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                task_id = request_task_id(source_conversation_id, request_id)
            except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
                raise TaskOrchestratorError("invalid_request", "invalid create request payload") from exc
        with self.create_lock if request_id is not None else nullcontext():
            if task_id is not None:
                try:
                    existing = self.store.get(task_id)
                except TaskStoreError:
                    existing = None
                if existing is not None:
                    if existing.get("create_request_fingerprint") != fingerprint:
                        raise TaskOrchestratorError("request_conflict", "create request id has a different payload")
                    return self.public(existing)
            try:
                plan = (self.planner.plan(utterance, timezone=timezone) if proposal is None else
                        self.planner.validate(proposal, utterance=utterance.strip(), now=self.clock()))
            except PlanError as exc:
                raise TaskOrchestratorError("invalid_plan", str(exc)) from exc
            now = self.clock()
            task_id = task_id or "task_" + uuid.uuid4().hex
            steps = [{**step, "index": index, "state": "pending"}
                     for index, step in enumerate(plan["steps"])]
            task = {
                "task_id": task_id, "state": "scheduled", "utterance": utterance.strip(),
                "summary": plan["summary"], "notification": plan["notification"],
                "schedule_kind": plan["schedule"]["kind"], "run_at": plan["schedule"]["run_at"],
                "timezone": timezone,
                "source_conversation_id": source_conversation_id,
                "steps": steps, "current_step": None, "cancel_requested": False,
                "created_at": now, "updated_at": now, "started_at": None, "ended_at": None,
                "error": None,
            }
            if fingerprint is not None:
                task["create_request_fingerprint"] = fingerprint
            try:
                created = self.store.create(task)
            except TaskStoreError as exc:
                raise TaskOrchestratorError("store_error", str(exc)) from exc
            self._event("task.created", created)
            self._event("task.scheduled", created)
            with self.condition:
                self.condition.notify_all()
            return self.public(created)

    def get(self, task_id: str) -> dict:
        try:
            return self.public(self.store.get(task_id))
        except TaskStoreError as exc:
            raise TaskOrchestratorError("not_found", str(exc)) from exc

    def list(self, *, states=None, limit=64) -> dict:
        try:
            items = self.store.list(states=states, limit=limit)
        except TaskStoreError as exc:
            raise TaskOrchestratorError("invalid_request", str(exc)) from exc
        return {"items": [self.public(item) for item in items]}

    def cancel(self, task_id: str) -> dict:
        with self.cancel_lock:
            try:
                task = self.store.get(task_id)
            except TaskStoreError as exc:
                raise TaskOrchestratorError("not_found", str(exc)) from exc
            if task["state"] == "completed" and self._skill_step(task):
                if task.get("skill_state") in {"stopped", "inactive"}:
                    return self.public(task)
                response = self._stop_skill(task)
                state = "stopped" if response.get("result", {}).get("stopped") else "inactive"
                task = self.store.update(task_id, skill_state=state, skill_stopped_at=self.clock())
                self._event("task.skill.stopped" if state == "stopped" else "task.skill.inactive", task)
                return self.public(task)
            if task["state"] in TERMINAL_STATES:
                return self.public(task)
            now = self.clock()
            if task["state"] == "scheduled":
                task = self.store.update(task_id, state="cancelled", cancel_requested=True, ended_at=now)
                self._event("task.cancelled", task)
            else:
                task = self.store.update(task_id, state="cancelling", cancel_requested=True)
                self._event("task.cancelling", task)
            with self.condition:
                self.condition.notify_all()
            return self.public(task)

    @staticmethod
    def _skill_step(task: dict) -> dict | None:
        return next((step for step in task.get("steps", []) if "skill" in step), None)

    def _stop_skill(self, task: dict) -> dict:
        step = self._skill_step(task)
        if step is None or self.skills is None:
            raise TaskOrchestratorError("skill_unavailable", "Skill stop unavailable")
        try:
            response = self.skills.stop(
                step["skill"], task_id=task["task_id"],
                request_id=f"{task['task_id']}-stop",
            )
        except Exception as exc:
            raise TaskOrchestratorError("skill_stop_failed", "Skill stop unavailable") from exc
        if (not isinstance(response, dict) or not response.get("ok")
                or not isinstance(response.get("result"), dict)):
            raise TaskOrchestratorError("skill_stop_failed", "Skill stop failed")
        return response

    def run_due(self, *, now=None) -> dict:
        if not self.execution_lock.acquire(blocking=False):
            return {"executed": [], "busy": True}
        executed = []
        try:
            for task in self.store.due(self.clock() if now is None else now):
                executed.append(self._execute(task["task_id"])["task_id"])
            return {"executed": executed, "busy": False}
        finally:
            self.execution_lock.release()

    def stats(self) -> dict:
        value = self.store.stats()
        value.update({"scheduler_alive": bool(self.thread and self.thread.is_alive()),
                      "execution_busy": self.execution_lock.locked()})
        return value

    def _execute(self, task_id: str) -> dict:
        task = self.store.get(task_id)
        if task["state"] != "scheduled":
            return task
        task = self.store.update(task_id, state="running", started_at=self.clock())
        self._event("task.started", task)
        try:
            for index, step in enumerate(task["steps"]):
                current = self.store.get(task_id)
                if current.get("cancel_requested"):
                    cancelled = self.store.update(task_id, state="cancelled", ended_at=self.clock(),
                                                  current_step=None)
                    self._event("task.cancelled", cancelled)
                    return cancelled
                steps = deepcopy(current["steps"])
                steps[index]["state"] = "running"
                steps[index]["started_at"] = self.clock()
                current = self.store.update(task_id, steps=steps, current_step=index)
                self._event("task.step.started", current, step=index)
                if "skill" in step:
                    if self.skills is None:
                        raise TaskOrchestratorError("skill_unavailable", "Skill executor unavailable")
                    response = self.skills.call(
                        step["skill"], step["arguments"], task_id=task_id,
                        request_id=f"{task_id}-step-{index}",
                    )
                else:
                    timeout_ms = self.planner.catalog.timeout_ms(step["capability"])
                    response = self.runtime.call(
                        step["capability"], step["arguments"], timeout_ms=timeout_ms,
                        request_id=f"{task_id}-step-{index}",
                    )
                if not isinstance(response, dict) or not response.get("ok"):
                    error = (response or {}).get("error") if isinstance(response, dict) else None
                    raise TaskOrchestratorError(
                        str((error or {}).get("code") or "runtime_failed"),
                        str((error or {}).get("message") or "Runtime capability failed"),
                    )
                if "skill" in step and not (response.get("result") or {}).get("accepted"):
                    raise TaskOrchestratorError("skill_not_admitted", "Skill admission was rejected")
                if self.store.get(task_id).get("cancel_requested"):
                    if "skill" in step:
                        self._stop_skill(task)
                    cancelled = self.store.update(task_id, state="cancelled", ended_at=self.clock(),
                                                  current_step=None)
                    self._event("task.cancelled", cancelled)
                    return cancelled
                current = self.store.get(task_id)
                steps = deepcopy(current["steps"])
                steps[index].update({
                    "state": "completed", "ended_at": self.clock(),
                    "result": _bounded_result(response.get("result")),
                })
                current = self.store.update(task_id, steps=steps, current_step=None)
                self._event("task.step.completed", current, step=index)
            with self.cancel_lock:
                if self.store.get(task_id).get("cancel_requested"):
                    if self._skill_step(task):
                        self._stop_skill(task)
                    cancelled = self.store.update(task_id, state="cancelled", ended_at=self.clock(),
                                                  current_step=None)
                    self._event("task.cancelled", cancelled)
                    return cancelled
                completed = self.store.update(
                    task_id, state="completed", ended_at=self.clock(), current_step=None,
                    **({"skill_state": "admitted"} if self._skill_step(task) else {}),
                )
            self._event("task.completed", completed)
            self._event("task.notification.requested", completed)
            return completed
        except Exception as exc:
            current = self.store.get(task_id)
            index = current.get("current_step")
            steps = deepcopy(current["steps"])
            if isinstance(index, int) and 0 <= index < len(steps):
                steps[index].update({"state": "failed", "ended_at": self.clock()})
            code = exc.code if isinstance(exc, TaskOrchestratorError) else "execution_failed"
            failed = self.store.update(
                task_id, state="failed", steps=steps, current_step=None, ended_at=self.clock(),
                error={"code": code, "message": str(exc)[:240]},
            )
            self._event("task.step.failed", failed, step=index)
            self._event("task.failed", failed)
            return failed

    def _scheduler_loop(self):
        while not self.stop_event.is_set():
            try:
                self.run_due()
            except Exception:
                LOG.exception("task scheduler iteration failed")
            next_run = self.store.next_run_at()
            wait = 30.0 if next_run is None else max(0.05, min(30.0, next_run - self.clock()))
            with self.condition:
                self.condition.wait(timeout=wait)

    def _event(self, topic: str, task: dict, *, step=None):
        payload = {
            "task_id": task["task_id"], "state": task["state"],
            "summary": task["summary"], "notification": task["notification"],
            "run_at": task["run_at"], "step": step,
            "task_kind": (
                "skill" if any("skill" in item for item in task["steps"])
                else "capability" if task["steps"] else "reminder"
            ),
        }
        if task.get("source_conversation_id"):
            payload["source_conversation_id"] = task["source_conversation_id"]
        if step is not None and isinstance(step, int) and 0 <= step < len(task["steps"]):
            if "capability" in task["steps"][step]:
                payload["capability"] = task["steps"][step]["capability"]
            if "skill" in task["steps"][step]:
                payload["skill"] = task["steps"][step]["skill"]
            payload["step_label"] = task["steps"][step]["label"]
        if task.get("error"):
            payload["error"] = task["error"]
        try:
            self.events.publish(topic, "task-orchestrator", payload,
                                privacy="local-private", trace_id=task["task_id"])
        except Exception as exc:
            LOG.warning("task event publish failed topic=%s error=%s", topic, type(exc).__name__)

    @staticmethod
    def public(task: dict) -> dict:
        value = deepcopy(task)
        value.pop("utterance", None)
        value.pop("create_request_fingerprint", None)
        return value


def _bounded_result(value):
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return {"available": False}
    if len(encoded.encode("utf-8")) <= 4096:
        return value
    return {"available": True, "truncated": True, "bytes": len(encoded.encode("utf-8"))}
