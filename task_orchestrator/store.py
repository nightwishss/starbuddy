"""Atomic bounded task persistence for the local task orchestrator."""

from __future__ import annotations

import json
import os
import threading
import time
from copy import deepcopy
from pathlib import Path


TERMINAL_STATES = {"completed", "failed", "cancelled"}
ACTIVE_STATES = {"scheduled", "running", "cancelling"}


class TaskStoreError(ValueError):
    pass


class TaskStore:
    def __init__(self, path: Path, *, clock=time.time, capacity=256):
        self.path = Path(path)
        self.clock = clock
        self.capacity = capacity
        self.lock = threading.RLock()
        self.tasks = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise TaskStoreError("task store is invalid") from exc
        if not isinstance(value, dict) or value.get("version") != 1 or not isinstance(value.get("tasks"), list):
            raise TaskStoreError("task store format is invalid")
        recovered = False
        for task in value["tasks"]:
            if isinstance(task, dict) and isinstance(task.get("task_id"), str):
                if task.get("state") in {"running", "cancelling"}:
                    task["state"] = "failed"
                    task["error"] = {"code": "interrupted", "message": "service restarted during execution"}
                    task["ended_at"] = self.clock()
                    recovered = True
                self.tasks[task["task_id"]] = task
        if recovered:
            self._write()

    def create(self, task: dict) -> dict:
        with self.lock:
            task_id = task.get("task_id")
            if not isinstance(task_id, str) or task_id in self.tasks:
                raise TaskStoreError("invalid or duplicate task_id")
            self._prune()
            if len(self.tasks) >= self.capacity:
                raise TaskStoreError("task capacity reached")
            self.tasks[task_id] = deepcopy(task)
            self._write()
            return deepcopy(task)

    def get(self, task_id: str) -> dict:
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                raise TaskStoreError("task not found")
            return deepcopy(task)

    def list(self, *, states=None, limit=64) -> list[dict]:
        if not isinstance(limit, int) or not 1 <= limit <= 256:
            raise TaskStoreError("invalid limit")
        selected = set(states or [])
        if selected - (TERMINAL_STATES | ACTIVE_STATES):
            raise TaskStoreError("invalid states")
        with self.lock:
            tasks = sorted(self.tasks.values(), key=lambda item: item.get("created_at", 0), reverse=True)
            if selected:
                tasks = [task for task in tasks if task.get("state") in selected]
            return deepcopy(tasks[:limit])

    def update(self, task_id: str, **changes) -> dict:
        with self.lock:
            if task_id not in self.tasks:
                raise TaskStoreError("task not found")
            self.tasks[task_id].update(deepcopy(changes))
            self.tasks[task_id]["updated_at"] = self.clock()
            self._write()
            return deepcopy(self.tasks[task_id])

    def due(self, now=None) -> list[dict]:
        timestamp = self.clock() if now is None else now
        with self.lock:
            values = [task for task in self.tasks.values()
                      if task.get("state") == "scheduled" and task.get("run_at", float("inf")) <= timestamp]
            return deepcopy(sorted(values, key=lambda item: (item["run_at"], item["created_at"])))

    def next_run_at(self):
        with self.lock:
            values = [task["run_at"] for task in self.tasks.values() if task.get("state") == "scheduled"]
            return min(values) if values else None

    def stats(self) -> dict:
        with self.lock:
            counts = {}
            for task in self.tasks.values():
                counts[task.get("state", "unknown")] = counts.get(task.get("state", "unknown"), 0) + 1
            return {"count": len(self.tasks), "capacity": self.capacity, "states": counts,
                    "persistence": "atomic-json", "path": str(self.path)}

    def _prune(self):
        if len(self.tasks) < self.capacity:
            return
        terminal = sorted((task for task in self.tasks.values() if task.get("state") in TERMINAL_STATES),
                          key=lambda item: item.get("ended_at", item.get("created_at", 0)))
        while terminal and len(self.tasks) >= self.capacity:
            self.tasks.pop(terminal.pop(0)["task_id"], None)

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        body = json.dumps({"version": 1, "tasks": list(self.tasks.values())},
                          ensure_ascii=False, separators=(",", ":"))
        temporary.write_text(body, encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)
