"""One durable programme receipt. Task schedules; its event requests delivery.

No polling thread or second scheduler. Claim before starting: crash recovery
prefers silence to unexpected replay. Receipt contains no raw browser token.
"""
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from gateway.display.quiet import PROGRAMS, QuietError, _owner
from gateway.speech.tempo import validate_speed
from task_orchestrator.client import request_task_id

PENDING = {"submitting", "scheduled"}


class QuietSchedule:
    def __init__(self, player, tasks, path, *, clock=time.time):
        self.player, self.tasks, self.path, self.clock = player, tasks, Path(path), clock
        self.lock = threading.RLock()
        self.record, self.broken = None, False
        try:
            if self.path.exists():
                if self.path.stat().st_size > 4096:
                    raise ValueError("oversized receipt")
                value = json.loads(self.path.read_text())
                if (not isinstance(value, dict) or value.get("version") != 1
                        or set(value) != {"version", "owner", "source", "task_id", "program_id", "volume_percent", "speed", "run_at", "state"}
                        or not isinstance(value["owner"], str) or not re.fullmatch(r"[a-f0-9]{64}", value["owner"])
                        or value["source"] != "quiet-programme:" + value["owner"][:32]
                        or not isinstance(value["task_id"], str) or not re.fullmatch(r"task_[a-f0-9]{32}", value["task_id"])
                        or value["program_id"] not in PROGRAMS
                        or type(value["volume_percent"]) is not int or not 0 <= value["volume_percent"] <= 60
                        or type(value["run_at"]) not in (int, float) or not 0 < value["run_at"] < 1e12
                        or value["state"] not in PENDING | {"missed", "cancelled", "claimed", "started", "skipped_busy"}):
                    raise ValueError("invalid receipt")
                validate_speed(value["speed"])
                self.record = value
        except (OSError, ValueError, TypeError):
            self.broken = True

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.path.with_suffix(".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(self.record, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)

    def status(self, owner):
        key = _owner(owner)
        with self.lock:
            r = self.record
            if r and r["state"] in PENDING and self.clock() > r["run_at"] + 60:
                r["state"] = "missed"
                self._save()
            return {"schedule_available": self.tasks is not None and not self.broken,
                    "schedule_occupied": bool(r and r["state"] in PENDING),
                    "schedule": ({k: r[k] for k in ("task_id", "program_id", "run_at", "state", "speed", "volume_percent")}
                                 if r and r["owner"] == key else None)}

    def create(self, owner, program_id, volume_percent, speed, run_at, request_id):
        key = _owner(owner)
        validate_speed(speed)
        if (not isinstance(program_id, str) or program_id not in PROGRAMS or type(volume_percent) is not int
                or not 0 <= volume_percent <= 60 or type(run_at) not in (int, float)
                or not isinstance(request_id, str) or not re.fullmatch(r"[a-f0-9]{32}", request_id)):
            raise ValueError("invalid schedule")
        source = "quiet-programme:" + key[:32]
        task_id = request_task_id(source, request_id)
        with self.lock:
            if self.tasks is None or self.broken:
                raise QuietError("预约服务未就绪")
            r = self.record
            if r and r["task_id"] == task_id:
                if any(r[k] != v for k, v in {"program_id": program_id, "volume_percent": volume_percent,
                                             "speed": speed, "run_at": run_at, "owner": key}.items()):
                    raise QuietError("预约重试参数已改变")
                if r["state"] != "submitting":
                    return self.status(owner)
            else:
                if not 30 <= run_at - self.clock() <= 86400:
                    raise ValueError("schedule must be 30 seconds..24 hours ahead")
                if r and r["state"] in PENDING:
                    raise QuietError("已有一个预约，请先取消")
                self.record = r = dict(version=1, owner=key, source=source, task_id=task_id,
                    program_id=program_id, volume_percent=volume_percent,
                    speed=speed, run_at=run_at, state="submitting")
                self._save()
            title = PROGRAMS[program_id]["title"]
            try:
                task = self.tasks.create("预约本地节目：" + title, timezone="UTC",
                    source_conversation_id=source, request_id=request_id,
                    proposal={"summary": "预约节目：" + title, "notification": "节目预约已到时间。",
                              "steps": [], "schedule": {"kind": "at", "run_at":
                                  datetime.fromtimestamp(run_at, timezone.utc).isoformat()}})
                if task.get("task_id") != task_id:
                    raise ValueError("unexpected task")
            except Exception as exc:
                raise QuietError("预约提交未确认；请刷新或取消，不要重复建立预约") from exc
            r["state"] = "scheduled"
            self._save()
            return self.status(owner)

    def cancel(self, owner, task_id):
        key = _owner(owner)
        with self.lock:
            r = self.record
            if not r or r["owner"] != key or r["task_id"] != task_id:
                raise QuietError("预约不属于当前页面")
            if r["state"] in {"claimed", "started"}:
                with self.player.lock:
                    if self.player.owner == key and self.player.scheduled:
                        self.player.control(owner, self.player.session, self.player.revision, "stop")
                r["state"] = "cancelled"
                self._save()
            elif r["state"] in PENDING:
                r["state"] = "cancelled"
                self._save()  # Revoke local permission before contacting Task.
            try:
                self.tasks.cancel(task_id)
            except Exception:
                pass  # No generic notification fallback; local revocation is durable.
            return self.status(owner)

    def deliver(self, task_id):
        with self.lock:
            r = self.record
            if self.broken or not r or r["task_id"] != task_id or r["state"] not in PENDING:
                return {"ok": True, "delivery": "ignored"}
            if not 0 <= self.clock() - r["run_at"] <= 60:
                r["state"] = "missed"
            else:
                task = self.tasks.get(task_id)  # Event is a signal, not authorization.
                if (task.get("state") != "completed" or task.get("steps") != []
                        or task.get("source_conversation_id") != r["source"]
                        or abs(task.get("run_at", 0) - r["run_at"]) > .001):
                    raise QuietError("预约任务尚未确认")
                r["state"] = "claimed"
            self._save()
            if r["state"] == "claimed":
                try:
                    if self.player.busy():
                        raise QuietError("foreground busy")
                    self.player._start(r["owner"], r["program_id"], r["volume_percent"], r["speed"], scheduled=True)
                    r["state"] = "started"
                except QuietError:
                    r["state"] = "skipped_busy"
                self._save()
            return {"ok": True, "delivery": r["state"]}
