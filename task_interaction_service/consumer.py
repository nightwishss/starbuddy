"""Bounded task-event consumer for StarBuddy display and audio output."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import tempfile
import time
from pathlib import Path
from observability.faults import FaultLog
from gateway.hugs import HugInbox, HugDelivery, TOPIC as HUG_TOPIC, ID as HUG_ID


LOG = logging.getLogger("vq.task-interaction")
TOPICS = [
    "task.scheduled", "task.started", "task.completed",
    "task.notification.requested", "task.failed",
    HUG_TOPIC,
]
TERMINAL_OUTPUT_STATES = {"completed", "failed", "cancelled", "expired", "interrupted"}


class TaskEventConsumer:
    def __init__(self, events, display, output, state_path: Path, *,
                 sleep=time.sleep, max_output_attempts=3, output_poll_seconds=0.25,
                 output_wait_seconds=90.0, hugs=None):
        self.events = events
        self.display = display
        self.output = output
        self.hugs = hugs if hugs is not None else HugDelivery(HugInbox(), display, output, sleep=sleep)
        self.state_path = Path(state_path)
        self.sleep = sleep
        self.max_output_attempts = max_output_attempts
        self.output_poll_seconds = output_poll_seconds
        self.output_wait_seconds = output_wait_seconds
        self.stop_event = threading.Event()
        self.state = self._load()
        self.faults = FaultLog(LOG)
        self._last_gap_log = float("-inf")

    def run_forever(self):
        failures = 0
        while not self.stop_event.is_set():
            try:
                self.poll_once(timeout=20.0)
                if failures:
                    LOG.info("task_event_recovered failures=%d", failures)
                failures = 0
            except Exception as exc:
                failures += 1
                delay = min(30.0, 2.0 ** min(failures, 5))
                LOG.warning("task_event_failed error=%s failures=%d retry_seconds=%.1f",
                            type(exc).__name__, failures, delay)
                self.stop_event.wait(delay)

    def poll_once(self, *, timeout=20.0) -> dict:
        result = self.events.wait(
            self.state["revision"], topics=TOPICS, timeout=timeout, limit=64,
        )
        if result.get("gap") and time.monotonic() - self._last_gap_log >= 60:
            self._last_gap_log = time.monotonic()
            LOG.warning("task_event_gap after_revision=%s oldest=%s latest=%s action=drain_retained",
                        self.state["revision"], result.get("oldest_revision"), result.get("revision"))
        events = result.get("events", [])
        for event in events:
            try:
                self._validate_event(event)
            except ValueError as exc:
                # Only malformed proposal data is rejected. Storage/handler failures
                # must propagate to bounded backoff WITHOUT acknowledging delivery.
                self.faults.record("task_event_rejected", action="skip_invalid", error=exc)
            else:
                self.handle(event)
            self._checkpoint(max(self.state["revision"], event["revision"]))
        latest = result.get("revision", self.state["revision"])
        if not events and latest < self.state["revision"]:
            # Event service restarted; semantic delivery keys still prevent duplicates.
            self._checkpoint(0)
        else:
            # New bus supplies the safe page boundary; support the original bus too.
            # A full legacy page may have unseen matches after its last event.
            boundary = result.get("next_revision", events[-1]["revision"] if len(events) >= 64 else latest)
            self._checkpoint(max(self.state["revision"], boundary))
        return result

    def _checkpoint(self, revision):
        previous = self.state["revision"]
        if previous == revision:
            return
        self.state["revision"] = revision
        try:
            self._save()
        except Exception:
            self.state["revision"] = previous
            raise

    def stop(self):
        self.stop_event.set()

    def handle(self, event: dict):
        topic = event.get("topic")
        payload = self._validate_event(event)
        if topic == HUG_TOPIC:
            self.hugs.deliver(payload["hug_id"])
            return
        task_id = payload["task_id"]
        source = payload.get("source_conversation_id", "")
        if isinstance(source, str) and re.fullmatch(r"quiet-programme:[a-f0-9]{32}", source):
            # Display revalidates the durable task and owns idempotent delivery.
            # Never emit a second generic/louder reminder for this programme.
            if topic == "task.notification.requested":
                key = "quiet:" + task_id
                record = self.state["deliveries"].get(key, {"attempts": 0})
                if record.get("state") in {"completed", "failed"}:
                    return
                for attempt in range(record["attempts"] + 1, 4):
                    record = {"attempts": attempt, "state": "attempting", "updated_at": time.time()}
                    self.state["deliveries"][key] = record
                    self._save()
                    try:
                        result = self.display.quiet_task(task_id)
                    except Exception as exc:
                        self.faults.record("quiet_delivery_unavailable", action="bounded_retry", error=exc)
                        if attempt < 3:
                            self.sleep(attempt)
                    else:
                        record.update(state="completed", delivery=result.get("delivery", "unknown"))
                        break
                if record.get("state") != "completed":
                    record["state"] = "failed"
                self.state["deliveries"][key] = record
                self._save()
            return
        trace_id = event.get("trace_id") or task_id
        if topic == "task.scheduled":
            self._display(phase="task_scheduled", visual_style="gentle",
                          primary_text=("技能任务已经安排好" if payload.get("task_kind") == "skill"
                                        else "提醒已经安排好"),
                          secondary_text=self._schedule_copy(payload), trace_id=trace_id)
        elif topic == "task.started":
            self._display(phase="acting", visual_style="attentive",
                          primary_text=("正在启动坐姿守护" if payload.get("task_kind") == "skill"
                                        else "提醒时间到了"),
                          secondary_text=(payload["summary"] if payload.get("task_kind") == "skill"
                                          else "我来告诉你"),
                          trace_id=trace_id)
        elif topic == "task.failed":
            self._display(phase="error", visual_style="concerned",
                          primary_text="这个任务没有完成",
                          secondary_text="你可以稍后再试一次", trace_id=trace_id)
            self.sleep(1.0)
            self._idle(trace_id)
        elif topic == "task.completed":
            self._display(phase="success", visual_style="gentle",
                          primary_text=("坐姿守护已经启动" if payload.get("task_kind") == "skill"
                                        else "任务已经完成"),
                          secondary_text=payload["summary"],
                          trace_id=trace_id)
        elif topic == "task.notification.requested":
            self._notification(payload, trace_id)

    def _notification(self, payload: dict, trace_id: str):
        task_id, text = payload["task_id"], payload["notification"]
        key = f"task.notification.requested:{task_id}"
        record = self.state["deliveries"].get(key)
        if record and record.get("state") in {"attempting", "accepted", "completed", "failed"}:
            LOG.info("duplicate task notification ignored task_id=%s", task_id)
            return
        self._display(phase="attention", visual_style="attentive",
                      primary_text="提醒你一下", secondary_text=text, trace_id=trace_id)
        job = None
        for attempt in range(1, self.max_output_attempts + 1):
            self.state["deliveries"][key] = {
                "state": "attempting", "attempts": attempt, "updated_at": time.time(),
            }
            self._save()
            try:
                job = self.output.say(
                    text, f"task:{task_id}", category="alert", volume_percent=75,
                    expires_in_seconds=120, trace_id=trace_id,
                )
            except Exception as exc:
                LOG.warning("task output admission failed task_id=%s attempt=%s error=%s",
                            task_id, attempt, type(exc).__name__)
                # A timeout after send is ambiguous: retrying could speak twice. Only
                # retry errors that prove no Unix-socket connection was established.
                safe_to_retry = isinstance(exc, (FileNotFoundError, ConnectionRefusedError))
                if safe_to_retry and attempt < self.max_output_attempts:
                    self.sleep(min(2 ** attempt, 8))
                else:
                    break
            else:
                self.state["deliveries"][key].update({
                    "state": "accepted", "job_id": job["job_id"], "updated_at": time.time(),
                })
                # Persistence failure is not an output admission failure.
                self._save()
                break
        if job is None:
            self.state["deliveries"][key]["state"] = "failed"
            self._save()
            # The persisted task remains complete; visual output may still be available.
            self.sleep(1.0)
            self._idle(trace_id)
            return
        terminal = self._wait_output(
            job["job_id"], on_playing=lambda: self._display(
                phase="speaking", visual_style="speaking",
                primary_text="提醒你一下", secondary_text=text, trace_id=trace_id,
            ),
        )
        self.state["deliveries"][key]["output_state"] = terminal or "unknown"
        if terminal == "completed":
            self.state["deliveries"][key]["state"] = "completed"
            self._display(phase="success", visual_style="gentle",
                          primary_text="提醒完成", secondary_text=text, trace_id=trace_id)
            self.sleep(1.0)
        elif terminal in TERMINAL_OUTPUT_STATES:
            self.state["deliveries"][key]["state"] = "failed"
            LOG.warning("task_output_finished task_id=%s state=%s action=no_replay", task_id, terminal)
        else:
            self.faults.record("task_output_unconfirmed", action="no_replay")
        self._save()
        self._idle(trace_id)

    def _idle(self, trace_id: str):
        self._display(phase="idle", visual_style="calm",
                      primary_text="我在这里", secondary_text="有事随时叫我", trace_id=trace_id)

    def _wait_output(self, job_id: str, *, on_playing=None) -> str | None:
        deadline = time.monotonic() + self.output_wait_seconds
        playing_reported = False
        while time.monotonic() < deadline:
            try:
                state = self.output.status(job_id).get("state")
            except Exception:
                return None
            if state == "playing" and not playing_reported:
                playing_reported = True
                if on_playing is not None:
                    on_playing()
            if state in TERMINAL_OUTPUT_STATES:
                return state
            self.sleep(self.output_poll_seconds)
        return None

    @staticmethod
    def _validate_event(event: dict) -> dict:
        if not isinstance(event, dict) or event.get("topic") not in TOPICS:
            raise ValueError("unsupported task event")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("invalid task event payload")
        if event['topic'] == HUG_TOPIC:
            if set(payload) != {'hug_id'} or not isinstance(payload['hug_id'], str) or not HUG_ID.fullmatch(payload['hug_id']):
                raise ValueError("invalid hug event")
            return payload
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or not task_id.startswith("task_") or len(task_id) > 96:
            raise ValueError("invalid task id")
        for key in ("summary", "notification"):
            if not isinstance(payload.get(key), str) or not payload[key].strip() or len(payload[key]) > 240:
                raise ValueError(f"invalid {key}")
        run_at = payload.get("run_at")
        if isinstance(run_at, bool) or not isinstance(run_at, (int, float)) or not math.isfinite(run_at):
            raise ValueError("invalid run_at")
        if payload.get("task_kind", "reminder") not in {"reminder", "capability", "skill"}:
            raise ValueError("invalid task kind")
        return payload

    def _display(self, **changes):
        try:
            self.display.update(producer="task", **changes)
        except Exception as exc:
            self.faults.record("task_display_unavailable", action="continue_audio", error=exc)

    @staticmethod
    def _schedule_copy(payload: dict) -> str:
        delay = max(0, int(payload["run_at"] - time.time()))
        if delay >= 3600 and delay % 3600 < 60:
            return f"大约 {delay // 3600} 小时后"
        if delay >= 60:
            return f"大约 {max(1, round(delay / 60))} 分钟后"
        return "很快就会提醒你"

    def _load(self):
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            if (isinstance(value, dict) and type(value.get("revision")) is int
                    and value["revision"] >= 0 and isinstance(value.get("deliveries"), dict)
                    and all(isinstance(item, dict) for item in value["deliveries"].values())):
                return value
        except FileNotFoundError:
            return {"revision": 0, "deliveries": {}}
        # Do not silently erase dedup state and replay reminders after corruption.
        raise ValueError("invalid task interaction state; operator review required")

    def _save(self):
        # Keep bounded dedup state; oldest entries are enough for local reminder scale.
        if len(self.state["deliveries"]) > 256:
            ordered = sorted(self.state["deliveries"].items(),
                             key=lambda item: item[1].get("updated_at", 0), reverse=True)
            self.state["deliveries"] = dict(ordered[:256])
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            # NamedTemporaryFile creates owner-only state before writing, and
            # replace remains atomic. No per-event fsync on the interaction path.
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                             dir=self.state_path.parent, prefix=".task-state-") as stream:
                temporary = Path(stream.name)
                json.dump(self.state, stream, ensure_ascii=False, separators=(",", ":"))
            os.replace(temporary, self.state_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
