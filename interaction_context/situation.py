"""Bounded, process-local task situation; no I/O, models or personal memory.

This is a small, trusted-task workspace, not a general scene-understanding model.
The caller supplies explicit task selections and deterministically checked answer
outcomes. A ticket only fences asynchronous computation: it does NOT authorize
profile access, microphone capture, model calls, or audible output. Existing
Runtime/Output admission and cancellation checks remain mandatory.
"""

from collections import deque
from copy import deepcopy
from itertools import count
import math
import threading
import time


TASKS = (
    {"id": "math-apples-1", "title": "苹果加法", "prompt": "有 3 个苹果，又拿来 2 个，一共有几个苹果？",
     "answer": "5", "hint": "先数原来的 3 个，再接着数新来的 2 个。",
     "steps": ["原来有 3 个苹果。", "又拿来 2 个，就是再加 2。", "3 + 2 = 5，一共有 5 个苹果。"]},
    {"id": "pattern-even-1", "title": "找数字规律", "prompt": "2、4、6、8，下一个数字是什么？",
     "answer": "10", "hint": "看看相邻两个数字，每次增加了多少。",
     "steps": ["4 比 2 多 2，6 比 4 多 2。", "每次都增加 2。", "8 + 2 = 10，所以下一个是 10。"]},
    {"id": "reading-rain-1", "title": "读懂一句话", "prompt": "“下雨了，小雨出门前拿了一把伞。”小雨为什么要拿伞？",
     "answer": "因为下雨了，要用伞挡雨。", "hint": "再看看句子开头，天气发生了什么变化。",
     "steps": ["句子告诉我们正在下雨。", "伞可以帮助挡雨。", "所以小雨拿伞，是为了出门时挡雨。"]},
)

# Public catalogue data is JSON-friendly. Its accidental mutation must never
# rewrite the authoritative trusted task inserted into a reasoning packet.
_TASK_BY_ID = {task["id"]: deepcopy(task) for task in TASKS}
_NAMESPACES = count(1)
_INPUTS = frozenset({"text", "voice", "touch"})
_ACTIONS = frozenset({"respond", "answer", "clarify", "offer_help", "wait",
                      "acknowledge", "complete", "cancel"})
_INTENTS = frozenset({"task_help", "task_answer", "task_selection", "decline_help",
                      "help_task", "answer_task", "clarify", "inspect_scene", "chat",
                      "cancel", "social", "unknown"})
_REASONS = frozenset({"explicit_request", "task_selected", "answer_checked", "uncertain",
                      "missing_task", "offer_eligible", "user_declined", "cancelled",
                      "task_complete", "no_action", "model_decision"})


def _bounded_ttl(value, maximum, name):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 0 < value <= maximum):
        raise ValueError("%s must be finite and in (0, %s]" % (name, maximum))
    return float(value)


class SituationState:
    """One active task and one pending turn, guarded by a revision and a token.

    Reads never extend the session. Selecting a task starts a fresh task episode;
    explicit decline suppresses unsolicited offers for that episode, but does not
    prevent the user subsequently requesting a hint. No input text is retained.
    """

    def __init__(self, clock=time.monotonic, *, inactivity_ttl=900, ticket_ttl=60):
        self._clock = clock
        self._inactivity_ttl = _bounded_ttl(inactivity_ttl, 900, "inactivity_ttl")
        self._ticket_ttl = _bounded_ttl(ticket_ttl, 60, "ticket_ttl")
        self._lock = threading.RLock()
        self._namespace = next(_NAMESPACES)
        self._serial = 0
        self._revision = 0
        self._last_clock = None
        self._last_activity = None
        self._task_id = None
        self._pending = None
        self._offers_enabled = False  # Proactive speech is opt-in, not inferred consent.
        self._declined = False
        self._offered = False
        self._completed = False
        self._correct = 0
        self._incorrect = 0
        self._outcomes = deque(maxlen=8)
        self._last_decision = None
        self._reason = "no_task"

    def _reset_task(self, reason):
        self._task_id = None
        self._last_activity = None
        self._pending = None
        self._declined = self._offered = self._completed = False
        self._correct = self._incorrect = 0
        self._outcomes.clear()
        self._last_decision = None
        self._reason = reason

    def _now(self):
        now = self._clock()
        if type(now) not in (int, float) or not math.isfinite(now):
            # Invalid time cannot preserve a ticket whose lifetime is unknown.
            self._reset_task("clock_invalid")
            self._revision += 1
            raise ValueError("clock must return a finite monotonic number")
        if self._last_clock is not None and now < self._last_clock:
            self._reset_task("clock_reset")
            self._revision += 1
        self._last_clock = now
        return now

    def _expire(self, now):
        if (self._task_id is not None and self._last_activity is not None
                and now - self._last_activity >= self._inactivity_ttl):
            self._reset_task("task_expired")
            self._revision += 1
        elif self._pending is not None and now >= self._pending["expires_at"]:
            self._pending = None
            self._revision += 1
            self._reason = "turn_expired"

    def _touch(self, now):
        if self._task_id is not None:
            self._last_activity = now

    def _invalidate(self, reason):
        self._revision += 1
        self._pending = None
        self._reason = reason

    def _offers(self):
        if self._task_id is None:
            reason = "no_task"
        elif not self._offers_enabled:
            reason = "disabled"
        elif self._declined:
            reason = "user_declined"
        elif self._completed:
            reason = "task_complete"
        elif self._offered:
            reason = "already_offered"
        elif self._incorrect < 2:
            reason = "insufficient_attempts"
        else:
            reason = "eligible"
        return {"enabled": self._offers_enabled, "declined": self._declined,
                "offered": self._offered, "eligible": reason == "eligible", "reason": reason}

    def _snapshot(self, now):
        task = _TASK_BY_ID.get(self._task_id)
        phase = "completed" if self._completed else "active" if task else "idle"
        return {"revision": self._revision,
                "task": {key: task[key] for key in ("id", "title", "prompt")} if task else None,
                "state": phase, "phase": phase,
                "reason": self._reason, "turn_pending": self._pending is not None,
                "attempts": {"correct": self._correct, "incorrect": self._incorrect,
                             "recent": list(self._outcomes)},
                "offers": self._offers(), "last_decision": deepcopy(self._last_decision),
                "decision": deepcopy(self._last_decision),
                "expires_in_s": max(0, round(self._inactivity_ttl - (now - self._last_activity), 3))
                if task and self._last_activity is not None else None}

    def select(self, task_id):
        if type(task_id) is not str or task_id not in _TASK_BY_ID:
            raise ValueError("unknown trusted task id")
        with self._lock:
            now = self._now()
            self._expire(now)
            self._reset_task("task_selected")
            self._task_id = task_id
            self._last_activity = now
            self._invalidate("task_selected")
            return self._snapshot(now)

    def clear(self):
        with self._lock:
            now = self._now()
            self._expire(now)
            self._reset_task("task_cleared")
            self._invalidate("task_cleared")
            return self._snapshot(now)

    def invalidate(self):
        """Cancel a pending turn while retaining the explicitly selected task."""
        with self._lock:
            now = self._now()
            self._expire(now)
            self._invalidate("cancelled")
            self._touch(now)
            return self._snapshot(now)

    def set_offers(self, enabled):
        if type(enabled) is not bool:
            raise TypeError("enabled must be a bool")
        with self._lock:
            now = self._now()
            self._expire(now)
            if enabled != self._offers_enabled:
                self._offers_enabled = enabled
                self._invalidate("offers_changed")
                self._touch(now)
            return self._snapshot(now)

    def begin(self, input_type):
        if type(input_type) is not str or input_type not in _INPUTS:
            raise ValueError("input_type must be text, voice or touch")
        with self._lock:
            now = self._now()
            self._expire(now)
            self._invalidate("turn_started")
            self._serial += 1
            self._touch(now)
            ticket = {"revision": self._revision, "task_id": self._task_id,
                      "token": "%d:%d" % (self._namespace, self._serial), "input_type": input_type}
            self._pending = {**ticket, "expires_at": now + self._ticket_ttl}
            return deepcopy(ticket)

    def _current(self, ticket):
        return bool(type(ticket) is dict and self._pending is not None
                    and type(ticket.get("revision")) is int
                    and all(ticket.get(key) == self._pending[key]
                            for key in ("revision", "task_id", "token", "input_type")))

    def current(self, ticket):
        with self._lock:
            now = self._now()
            self._expire(now)
            return self._current(ticket)

    def finish(self, ticket, decision):
        """Commit one structured result, never arbitrary text or profile fields.

        Success consumes the ticket. This method records a decision, not proof
        of execution/delivery; the caller retains responsibility for Output.
        """
        with self._lock:
            now = self._now()
            self._expire(now)
            if not self._current(ticket) or type(decision) is not dict:
                return False
            action = decision.get("action")
            if type(action) is not str or action not in _ACTIONS:
                return False
            if action == "offer_help" and not self._offers()["eligible"]:
                return False
            result = {"action": action}
            if type(decision.get("intent")) is str and decision["intent"] in _INTENTS:
                result["intent"] = decision["intent"]
            if type(decision.get("reason")) is str and decision["reason"] in _REASONS:
                result["reason"] = decision["reason"]
            if type(decision.get("needs_clarification")) is bool:
                result["needs_clarification"] = decision["needs_clarification"]
            if action == "offer_help":
                self._offered = True
            self._last_decision = result
            self._pending = None
            self._reason = "decision_recorded"
            self._touch(now)
            return True

    def feedback(self, correct, *, ticket=None):
        """Record a verified outcome, optionally fenced to its exact old task."""
        if type(correct) is not bool:
            raise TypeError("correct must be a bool")
        with self._lock:
            now = self._now()
            self._expire(now)
            if ticket is not None and not self._current(ticket):
                return None
            if self._task_id is not None and not self._completed:
                self._invalidate("answer_checked")
                self._outcomes.append(correct)
                if correct:
                    self._correct = min(999, self._correct + 1)
                    self._completed = True
                else:
                    self._incorrect = min(999, self._incorrect + 1)
                self._touch(now)
            return self._snapshot(now)

    def decline(self, *, ticket=None):
        """Explicitly decline unsolicited help for this task episode."""
        with self._lock:
            now = self._now()
            self._expire(now)
            if ticket is not None and not self._current(ticket):
                return None
            self._invalidate("user_declined")
            if self._task_id is not None:
                self._declined = True
            self._touch(now)
            return self._snapshot(now)

    def snapshot(self):
        with self._lock:
            now = self._now()
            self._expire(now)
            return self._snapshot(now)

    def packet(self, ticket):
        """Trusted, bounded task context; None for a cancelled/expired turn.

        A visible person is not evidence of the microphone speaker or the touch
        operator. This packet grants no private-profile access for either.
        """
        with self._lock:
            now = self._now()
            self._expire(now)
            if not self._current(ticket):
                return None
            unknown = {"identity": "unknown", "profile_authorized": False}
            return {"schema": "situated-task-v1", "revision": self._revision,
                    "input_type": self._pending["input_type"],
                    "task": deepcopy(_TASK_BY_ID.get(self._task_id)),
                    "task_source": "explicit_ui_selection" if self._task_id else None,
                    "state": "completed" if self._completed else "active" if self._task_id else "idle",
                    "attempts": {"correct": self._correct, "incorrect": self._incorrect,
                                 "recent": list(self._outcomes)},
                    "offers": self._offers(),
                    "actors": {"microphone_speaker": deepcopy(unknown),
                               "touch_operator": deepcopy(unknown),
                               "visible_person": {"identity": "not_supplied",
                                                  "profile_authorized": False},
                               "cross_modal_identity_link": "unproven"}}
