"""Typed public reminder actions and an ephemeral foreground work budget.

No model output is executable here. Only a small explicit grammar can produce a
confirmation draft; only Task Orchestrator receipts establish task state. There
is no inference, device access, profile lookup, list/history scan or persistence.
"""

from __future__ import annotations

from copy import deepcopy
import math
import re
import secrets
import threading
import time

from task_orchestrator.client import request_task_id


ACTION_SESSION = "starbuddy.situated.actions"
_REMOTE_STATES = {"scheduled", "running", "cancelling", "cancelled", "completed", "failed"}
_TERMINAL = {"cancelled", "completed", "failed"}
_TOPICS = {"休息": "rest", "休息一下": "rest", "喝水": "water",
           "起来活动": "move", "站起来活动": "move", "活动一下": "move"}
_CANONICAL = {"rest": ("休息", "该休息了"), "water": ("喝水", "该喝水了"),
              "move": ("起来活动", "该起来活动了")}
_NUM = r"(?:[1-9][0-9]{0,3}|[一二两三四五六七八九十百千零〇]{1,12})"
_PROPOSE = re.compile(r"(?:请)?(?:在)?(?P<number>" + _NUM + r")\s*分钟后提醒我(?P<topic>休息一下|站起来活动|起来活动|活动一下|休息|喝水)[。！!？?]?")
_MODIFY = re.compile(r"(?:请)?改成(?P<number>" + _NUM + r")\s*分钟[。！!？?]?")


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _chinese(number):
    digits = "零一二三四五六七八九"
    result = ""
    need_zero = False
    for index, char in enumerate(str(number)):
        place = len(str(number)) - index - 1
        if char == "0":
            need_zero = bool(result)
        else:
            if need_zero:
                result += "零"
            result += digits[int(char)] + ("", "十", "百", "千")[place]
            need_zero = False
    return result[1:] if result.startswith("一十") else result


def _minutes(text):
    if text.isascii() and text.isdigit():
        value = int(text)
    else:
        normalized = text.replace("两", "二").replace("〇", "零")
        total, digit = 0, 0
        for char in normalized:
            if char in "零一二三四五六七八九":
                digit = "零一二三四五六七八九".index(char)
            else:
                total += (digit or 1) * {"十": 10, "百": 100, "千": 1000}[char]
                digit = 0
        value = total + digit
        if not 1 <= value <= 1440 or _chinese(value) != normalized:
            return None
    return value if 1 <= value <= 1440 else None


def is_reminder_request(message):
    """Unsupported reminder wording must clarify, not become model execution."""
    return isinstance(message, str) and any(word in message for word in ("提醒", "定时", "倒计时", "改成"))


class ReminderActionBroker:
    """One owned task/draft. Unknown create outcomes are never retried.

    Cancellation during create records a pending cancel, not a false success.
    Reconciliation queries only the stable ID from this broker's create request.
    A 60-second deadline applies only to unconfirmed drafts, never to a task
    that may already exist in the persistent Task Orchestrator.
    """

    def __init__(self, task_client, *, clock=time.monotonic, session_id=ACTION_SESSION,
                 timezone="Europe/Madrid", refresh_interval=2.0):
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}", session_id):
            raise ValueError("invalid action session")
        if not isinstance(timezone, str) or not re.fullmatch(r"[A-Za-z_]+/[A-Za-z_]+(?:/[A-Za-z_]+)?", timezone):
            raise ValueError("invalid action timezone")
        if not _finite(refresh_interval) or not 1 <= refresh_interval <= 60:
            raise ValueError("invalid refresh interval")
        self.task_client = task_client
        self.clock = clock
        self.session_id = session_id
        self.timezone = timezone
        self.refresh_interval = refresh_interval
        self._lock = threading.RLock()
        self._record = None
        self._last_clock = None
        self._next_refresh = 0

    def _available(self):
        return all(callable(getattr(self.task_client, name, None)) for name in ("create", "get", "cancel"))

    def _now(self):
        try:
            now = self.clock()
        except Exception:
            now = None
        if not _finite(now) or self._last_clock is not None and now < self._last_clock:
            if self._record:
                self._record["status"] = "unknown" if self._record["claimed"] else "expired"
                self._record["reason"] = "clock_invalid"
            return None
        self._last_clock = now
        if (self._record and not self._record["claimed"] and self._record["status"] == "proposed"
                and now >= self._record["expires_at"]):
            self._record.update(status="expired", reason="draft_expired")
        return now

    def _public(self, now=None, *, reason=None, accepted=None):
        record = self._record
        if record is None:
            value = {"status": "idle", "state": "idle", "reason": reason or "no_action"}
        else:
            value = {"status": record["status"], "state": record["status"],
                     "reason": reason or record["reason"], "summary": record["summary"],
                     "minutes": record["minutes"], "topic": record["topic"],
                     "input_type": record["input_type"], "cancel_requested": record["cancel_requested"]}
            if record["status"] == "proposed":
                value.update(token=record["token"], expires_in_s=max(0, round(record["expires_at"] - now, 3)) if now is not None else 0)
            if record["receipt_known"]:
                value["task_id"] = record["task_id"]
        if accepted is not None:
            value["accepted"] = accepted
        return deepcopy(value)

    def pending(self):
        with self._lock:
            return self._public(self._now())

    def discard_draft(self, require_resettable=False):
        """Atomically revoke only an unclaimed draft; never cancel a real task.

        Navigation/attention/reset controls must not do a racy pending() check
        followed by cancel(), which could instead cancel a concurrently claimed
        real reminder. Reset may additionally refuse to forget unresolved work.
        """
        if type(require_resettable) is not bool:
            raise ValueError("resettable flag must be boolean")
        with self._lock:
            self._now()
            record = self._record
            if record and record["claimed"] and record["status"] not in _TERMINAL:
                if require_resettable:
                    raise RuntimeError("提醒任务仍在进行或结果未确认，请先处理该任务")
                return False
            if record and not record["claimed"] and record["status"] == "proposed":
                record.update(status="cancelled", reason="draft_discarded")
                return True
            return False

    def status(self):
        with self._lock:
            value = self._public(self._now())
            return {"status": value["status"], "reason": value["reason"],
                    "draft": value if value["status"] == "proposed" else None,
                    "task": {"task_id": value["task_id"], "state": value["status"]} if "task_id" in value else None,
                    "cancel_requested": bool(self._record and self._record["cancel_requested"]),
                    "available": self._available()}

    def propose(self, message, input_type="text"):
        if type(message) is not str or type(input_type) is not str or input_type not in {"text", "voice", "touch"}:
            return None
        text = message.strip()
        proposed = _PROPOSE.fullmatch(text) if 0 < len(text) <= 120 else None
        modified = _MODIFY.fullmatch(text) if 0 < len(text) <= 120 else None
        match = proposed or modified
        minutes = _minutes(match["number"]) if match else None
        with self._lock:
            now = self._now()
            if now is None:
                return None
            record = self._record
            if record and record["claimed"] and record["status"] not in _TERMINAL:
                return None
            if minutes is None or modified and (record is None or record["status"] != "proposed" or record["claimed"]):
                if record and not record["claimed"] and record["status"] == "proposed" and is_reminder_request(text):
                    record.update(status="cancelled", reason="unsupported_revision")
                return None
            topic = record["topic"] if modified else _TOPICS[proposed["topic"]]
            action, notification = _CANONICAL[topic]
            token = "action_" + secrets.token_hex(12)
            request_id = "situated-" + secrets.token_hex(12)
            self._record = {"token": token, "request_id": request_id,
                            "task_id": request_task_id(self.session_id, request_id),
                            "minutes": minutes, "topic": topic, "input_type": input_type,
                            "summary": f"{minutes}分钟后提醒：{notification}",
                            "canonical": f"{minutes}分钟后提醒我{action}",
                            "notification": notification, "expires_at": now + 60,
                            "status": "proposed", "reason": "confirmation_required", "claimed": False,
                            "receipt_known": False, "cancel_requested": False,
                            "cancel_dispatched": False, "rpc": None}
            self._next_refresh = 0
            return self._public(now)

    def _receipt(self, record, value):
        return (type(value) is dict and value.get("task_id") == record["task_id"]
                and value.get("source_conversation_id") == self.session_id
                and type(value.get("state")) is str and value["state"] in _REMOTE_STATES)

    def confirm(self, token):
        with self._lock:
            now = self._now()
            record = self._record
            if now is None or type(token) is not str or not record or token != record["token"]:
                return self._public(now, reason="stale_token", accepted=False)
            if record["claimed"]:
                return self._public(now, reason="already_submitted", accepted=False)
            if record["status"] != "proposed":
                return self._public(now, reason="draft_unavailable", accepted=False)
            if not self._available():
                record.update(status="failed", reason="task_service_unavailable")
                return self._public(now, accepted=False)
            record.update(claimed=True, status="submitting", reason="create_submitting", rpc="create")
            proposal = {"summary": record["summary"],
                        "schedule": {"kind": "delay", "delay_seconds": record["minutes"] * 60},
                        "notification": record["notification"], "steps": []}
        try:
            reply = self.task_client.create(record["canonical"], timezone=self.timezone,
                                            source_conversation_id=self.session_id,
                                            proposal=proposal, request_id=record["request_id"])
        except Exception:
            reply = None
        with self._lock:
            record["rpc"] = None
            valid = self._receipt(record, reply)
            if valid:
                record.update(status=reply["state"], reason="create_confirmed", receipt_known=True)
            else:
                record.update(status="unknown", reason="create_outcome_unknown")
            cancel = valid and record["cancel_requested"] and record["status"] not in _TERMINAL
            result = self._public(self._now(), accepted=valid)
        return self._dispatch_cancel(record) if cancel else result

    def _dispatch_cancel(self, record):
        with self._lock:
            if (record is not self._record or record["rpc"] is not None
                    or not record["receipt_known"] or record["cancel_dispatched"]):
                return self._public(self._now(), reason="cancel_outcome_unconfirmed", accepted=False)
            if record["status"] in _TERMINAL:
                return self._public(self._now(), reason="already_terminal", accepted=False)
            record.update(cancel_dispatched=True, rpc="cancel", status="cancelling", reason="cancel_submitting")
        try:
            reply = self.task_client.cancel(record["task_id"])
        except Exception:
            reply = None
        with self._lock:
            record["rpc"] = None
            valid = self._receipt(record, reply)
            if valid:
                record.update(status=reply["state"], reason="cancel_confirmed" if reply["state"] == "cancelled" else "cancel_not_terminal")
            else:
                record.update(status="unknown", reason="cancel_outcome_unknown")
            result = self._public(self._now(), accepted=valid)
            result["cancelled"] = valid and reply["state"] == "cancelled"
            return result

    def cancel(self, token=None):
        with self._lock:
            now = self._now()
            record = self._record
            if not record or token is not None and (type(token) is not str or token != record["token"]):
                return self._public(now, reason="stale_token", accepted=False)
            if not record["claimed"]:
                if record["status"] != "proposed":
                    return self._public(now, reason="draft_unavailable", accepted=False)
                record.update(status="cancelled", reason="draft_cancelled")
                return {**self._public(now, accepted=True), "cancelled": True}
            if record["status"] in _TERMINAL:
                # A repeated control must not downgrade an acknowledged
                # cancelled/completed/failed task to an invented unknown state.
                return {**self._public(now, reason="already_terminal", accepted=False),
                        "cancelled": record["status"] == "cancelled"}
            record["cancel_requested"] = True
            if record["rpc"] is not None:
                record.update(status="unknown", reason="cancel_waiting_for_receipt")
                return {**self._public(now, accepted=False), "cancelled": False}
            if not record["receipt_known"] or record["cancel_dispatched"]:
                record.update(status="unknown", reason="cancel_outcome_unconfirmed")
                return {**self._public(now, accepted=False), "cancelled": False}
        return self._dispatch_cancel(record)

    def refresh(self):
        with self._lock:
            now = self._now()
            record = self._record
            if (now is None or not record or not record["claimed"] or record["rpc"] is not None
                    or record["status"] in _TERMINAL or now < self._next_refresh or not self._available()):
                return self._public(now)
            self._next_refresh = now + self.refresh_interval
            record["rpc"] = "get"
        try:
            reply = self.task_client.get(record["task_id"])
        except Exception:
            reply = None
        with self._lock:
            record["rpc"] = None
            valid = self._receipt(record, reply)
            if valid:
                record.update(status=reply["state"], receipt_known=True, reason="status_confirmed")
            else:
                record.update(status="unknown", reason="status_unconfirmed")
            if (valid and record["cancel_requested"] and not record["cancel_dispatched"]
                    and record["status"] not in _TERMINAL):
                # GET/status and read-only MCP callers must never acquire a
                # deferred write side effect. A fresh explicit cancel action
                # can use this receipt to dispatch the still-pending cancel.
                record["reason"] = "cancel_requires_explicit_retry"
            return self._public(self._now())


class ForegroundBudget:
    """Nonblocking single-flight ownership; revocation is not native cancellation.

    Exactly one model claim and one inspection claim are allowed per ticket.
    Cancel/deadline revoke authority but keep the occupied slot until finish(),
    so a still-running native call cannot be overlapped by a replacement action.
    Resource busy gates claims only; it never gates explicit user controls.
    """

    def __init__(self, *, clock=time.monotonic, timeout_seconds=15.0):
        if not _finite(timeout_seconds) or not 0 < timeout_seconds <= 60:
            raise ValueError("invalid foreground deadline")
        self.clock = clock
        self.timeout_seconds = timeout_seconds
        self._lock = threading.RLock()
        self._generation = 0
        self._last_clock = None
        self._active = None
        self._last = {"model_calls": 0, "inspections": 0, "reason": "idle"}

    def _now(self):
        try:
            value = self.clock()
        except Exception:
            value = None
        if (not _finite(value) or self._last_clock is not None and value < self._last_clock):
            if self._active:
                self._active.update(cancelled=True, reason="clock_invalid_waiting_for_finish")
            return None
        self._last_clock = value
        return value

    def _matches(self, ticket):
        return (type(ticket) is dict and self._active is not None
                and type(ticket.get("generation")) is int
                and all(ticket.get(key) == self._active["ticket"][key] for key in ("generation", "token")))

    def _valid(self, ticket, now):
        active = self._active
        return bool(self._matches(ticket) and now is not None and not active["cancelled"]
                    and active["started"] <= now < active["deadline"])

    def begin(self, *, current=None, resource_busy=False):
        if current is not None:
            try:
                if not callable(current) or current() is not True:
                    return None
            except Exception:
                return None
        with self._lock:
            now = self._now()
            if self._active is not None or now is None:
                return None
            self._generation += 1
            ticket = {"generation": self._generation, "token": "foreground_" + secrets.token_hex(12)}
            self._active = {"ticket": ticket, "started": now, "deadline": now + self.timeout_seconds,
                            "cancelled": False, "current": current, "model_calls": 0, "inspections": 0,
                            "reason": "admitted"}
            return deepcopy(ticket)

    def current(self, ticket):
        with self._lock:
            if not self._valid(ticket, self._now()):
                return False
            callback = self._active["current"]
        if callback is not None:
            try:
                if callback() is not True:
                    return False
            except Exception:
                return False
        with self._lock:
            return self._valid(ticket, self._now())

    def claim(self, ticket, kind, *, resource_busy=False):
        field = {"model": "model_calls", "inspection": "inspections"}.get(kind) if type(kind) is str else None
        if field is None or not self.current(ticket):
            return False
        with self._lock:
            if not self._valid(ticket, self._now()):
                return False
            if type(resource_busy) is not bool or resource_busy:
                self._active["reason"] = "resource_busy"
                return False
            if self._active[field] >= 1:
                self._active["reason"] = "budget_exhausted"
                return False
            self._active[field] += 1
            self._active["reason"] = "claim_admitted"
            return True

    def cancel(self):
        with self._lock:
            if self._active is None:
                return False
            self._active.update(cancelled=True, reason="cancelled_waiting_for_finish")
            self._generation += 1
            return True

    def finish(self, ticket):
        with self._lock:
            if not self._matches(ticket):
                return False
            self._last = {key: self._active[key] for key in ("model_calls", "inspections", "reason")}
            self._active = None
            return True

    def status(self):
        with self._lock:
            active = self._active
            now = self._now()
            counts = active or self._last
            return {"busy": active is not None, "model_calls": counts["model_calls"],
                    "inspections": counts["inspections"], "reason": counts["reason"],
                    "cancelled": bool(active and active["cancelled"]),
                    "deadline_remaining_s": max(0, round(active["deadline"] - now, 3)) if active and now is not None else 0,
                    "limits": {"model_calls": 1, "inspections": 1}}
