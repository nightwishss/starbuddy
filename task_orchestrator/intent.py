"""Deterministic conversational intent routing for bounded reminder drafts."""

from __future__ import annotations

import re
import secrets
import threading
import time
from copy import deepcopy


MAX_DRAFTS = 16
DRAFT_TTL_SECONDS = 10 * 60
_UNCERTAIN_CREATION = "task creation outcome uncertain; retry confirmation or inspect task status"
_REMINDER_WORDS = ("提醒我", "提醒一下", "叫我", "通知我")
_TASK_WORDS = _REMINDER_WORDS + ("定时", "倒计时", "安排", "任务")
_POSTURE_WORDS = ("坐姿", "姿势", "驼背", "体态")
_MONITOR_WORDS = ("看着", "留意", "监督", "监测", "检查", "提醒")
_DELAY_RE = re.compile(
    r"(?P<number>\d{1,5}|[零〇一二两三四五六七八九十百千]+)\s*"
    r"(?P<unit>秒钟?|分钟?|小时|个小时)\s*(?:以后|之后|后)"
)
_DURATION_RE = re.compile(
    r"(?P<number>半|\d{1,4}|[零〇一二两三四五六七八九十百千]+)\s*"
    r"(?P<unit>分钟?|小时|个小时)"
)
_CHINESE_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3,
                   "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


class IntentError(ValueError):
    pass


def _chinese_integer(text: str) -> int:
    if text.isdigit():
        return int(text)
    if all(char in _CHINESE_DIGITS for char in text):
        return int("".join(str(_CHINESE_DIGITS[char]) for char in text))
    total, section, digit = 0, 0, 0
    units = {"十": 10, "百": 100, "千": 1000}
    for char in text:
        if char in _CHINESE_DIGITS:
            digit = _CHINESE_DIGITS[char]
        elif char in units:
            unit = units[char]
            section += (digit or 1) * unit
            digit = 0
        else:
            raise IntentError("invalid Chinese number")
    total += section + digit
    return total


def _notification_text(utterance: str, match: re.Match) -> str:
    tail = utterance[match.end():].strip(" ，,。.!！?？")
    for prefix in _REMINDER_WORDS:
        if tail.startswith(prefix):
            tail = tail[len(prefix):].strip(" ，,。.!！?？")
            break
    if not tail:
        tail = "你设定的提醒时间到了"
    if len(tail) > 180:
        raise IntentError("reminder text is too long")
    return tail if tail.endswith(("了", "啦", "。", "！", "!")) else f"该{tail}了"


class ReminderIntentRouter:
    """Classify clear reminders without granting the model execution authority."""

    def classify(self, utterance: str, *, timezone="UTC",
                 source_conversation_id: str | None = None) -> dict:
        if not isinstance(utterance, str) or not utterance.strip() or len(utterance) > 1000:
            raise IntentError("utterance must be 1..1000 characters")
        text = utterance.strip()
        posture = any(word in text for word in _POSTURE_WORDS)
        monitor = any(word in text for word in _MONITOR_WORDS)
        duration = _DURATION_RE.search(text)
        if posture and monitor and duration:
            if duration.group("number") == "半":
                seconds = 30 * 60 if "小时" in duration.group("unit") else None
            else:
                number = _chinese_integer(duration.group("number"))
                seconds = number * (3600 if "小时" in duration.group("unit") else 60)
            if seconds is None or not 60 <= seconds <= 4 * 60 * 60:
                return {"intent": "ambiguous", "reason": "duration_out_of_range"}
            steps = [{
                "skill": "monitor_posture",
                "arguments": {"duration_seconds": seconds},
                "label": "在限定时长内守护坐姿",
            }]
            # A deliberately narrow conversational composition: an explicitly
            # ordered matrix acknowledgement may precede the bounded Skill.
            # Other combinations go to clarification, never silent omission.
            if any(word in text for word in ("矩阵", "灯")):
                icons = {"爱心": "heart", "笑脸": "smile"}
                matches = [(word, name) for word, name in icons.items() if word in text]
                if (len(matches) != 1 or "先" not in text or "然后" not in text
                        or not text.index("先") < text.index(matches[0][0]) < text.index("然后")
                        or not any(word in text[text.index("然后"):] for word in _POSTURE_WORDS)):
                    return {"intent": "ambiguous", "reason": "unsupported_skill_pipeline"}
                steps.insert(0, {
                    "capability": "matrix.icon", "arguments": {"name": matches[0][1]},
                    "label": "先显示指定矩阵图标",
                })
            return {
                "intent": "task",
                "draft": {
                    "task_type": "skill", "summary": (
                        f"先显示矩阵图标，再持续 {round(seconds / 60)} 分钟坐姿守护"
                        if len(steps) > 1 else f"持续 {round(seconds / 60)} 分钟坐姿守护"
                    ),
                    "schedule": {"kind": "now"}, "timezone": timezone,
                    "notification": "好的，坐姿守护已经开始。",
                    "steps": steps,
                    "source_conversation_id": source_conversation_id,
                    "utterance": text,
                },
            }
        has_task_hint = any(word in text for word in _TASK_WORDS)
        delay = _DELAY_RE.search(text)
        if delay and any(word in text for word in _REMINDER_WORDS):
            number = _chinese_integer(delay.group("number"))
            multiplier = 1 if delay.group("unit").startswith("秒") else (
                60 if delay.group("unit").startswith("分") else 3600
            )
            seconds = number * multiplier
            if not 1 <= seconds <= 7 * 24 * 60 * 60:
                return {"intent": "ambiguous", "reason": "delay_out_of_range"}
            notification = _notification_text(text, delay)
            return {
                "intent": "task",
                "draft": {
                    "task_type": "reminder",
                    "summary": f"{delay.group(0)}提醒：{notification}"[:200],
                    "delay_seconds": seconds,
                    "schedule": {"kind": "delay", "delay_seconds": seconds},
                    "timezone": timezone,
                    "notification": notification,
                    "steps": [],
                    "source_conversation_id": source_conversation_id,
                    "utterance": text,
                },
            }
        if has_task_hint:
            return {"intent": "ambiguous", "reason": "missing_or_unsupported_schedule"}
        return {"intent": "chat"}


class TaskDraftStore:
    """Small ephemeral confirmation store; tasks are persisted only after confirmation."""

    def __init__(self, *, clock=time.time, ttl_seconds=DRAFT_TTL_SECONDS):
        self.clock = clock
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()
        self.values: dict[str, dict] = {}

    def create(self, draft: dict) -> dict:
        now = self.clock()
        value = deepcopy(draft)
        value.update({
            "draft_id": "draft_" + secrets.token_hex(12),
            "created_at": now, "expires_at": now + self.ttl_seconds,
        })
        if isinstance(value.get("_decision"), dict):
            value["expires_at"] = min(value["expires_at"], value["_decision"]["expires_at"])
        with self.lock:
            self._prune(now)
            source = value.get("source_conversation_id")
            if source:
                for item in self.values.values():
                    if item.get("source_conversation_id") == source:
                        self._require_removable(item)
                # A revised request supersedes any unconfirmed draft in this session.
                for key in [key for key, item in self.values.items()
                            if item.get("source_conversation_id") == source]:
                    self.values.pop(key, None)
            if len(self.values) >= MAX_DRAFTS:
                available = [key for key, item in self.values.items()
                             if not item.get("_confirming") and not item.get("_uncertain")]
                if not available:
                    if any(item.get("_uncertain") for item in self.values.values()):
                        raise IntentError(_UNCERTAIN_CREATION)
                    raise IntentError("all task drafts are being confirmed")
                oldest = min(available, key=lambda key: self.values[key]["created_at"])
                self.values.pop(oldest, None)
            self.values[value["draft_id"]] = value
        return self.public(value)

    def consume(self, draft_id: str) -> dict:
        if not isinstance(draft_id, str) or not draft_id.startswith("draft_"):
            raise IntentError("invalid draft id")
        now = self.clock()
        with self.lock:
            self._prune(now)
            value = self.values.get(draft_id)
            if value is not None:
                self._require_removable(value)
                self.values.pop(draft_id, None)
        if value is None:
            raise IntentError("task draft expired or not found")
        return value

    def get(self, draft_id: str, *, source_conversation_id: str | None = None) -> dict:
        if not isinstance(draft_id, str) or not draft_id.startswith("draft_"):
            raise IntentError("invalid draft id")
        now = self.clock()
        with self.lock:
            self._prune(now)
            value = self.values.get(draft_id)
            if value is None:
                raise IntentError("task draft expired or not found")
            if value.get("_confirming"):
                raise IntentError("task draft confirmation already in progress")
            if (source_conversation_id is not None
                    and value.get("source_conversation_id") != source_conversation_id):
                raise IntentError("task draft belongs to another session")
            return deepcopy(value)

    def claim(self, draft_id: str, *, source_conversation_id: str) -> dict:
        """Atomically reserve one draft so concurrent confirms cannot create twice."""
        now = self.clock()
        with self.lock:
            self._prune(now)
            value = self.values.get(draft_id)
            if value is None:
                raise IntentError("task draft expired or not found")
            if value.get("source_conversation_id") != source_conversation_id:
                raise IntentError("task draft belongs to another session")
            if value.get("_confirming"):
                raise IntentError("task draft confirmation already in progress")
            value["_confirming"] = True
            return deepcopy(value)

    def finish_claim(self, draft_id: str, *, success: bool, uncertain: bool = False) -> None:
        """Release a claim, retaining unknown outcomes until reconciled.

        An unsuccessful retry cannot erase an earlier uncertain outcome. A
        caller that establishes the outcome removes it with success=True.
        """
        with self.lock:
            value = self.values.get(draft_id)
            if value is None:
                return
            if success:
                self.values.pop(draft_id, None)
            else:
                value.pop("_confirming", None)
                if uncertain:
                    value["_uncertain"] = True

    def cancel(self, draft_id: str, *, source_conversation_id: str | None = None) -> bool:
        if not isinstance(draft_id, str) or not draft_id.startswith("draft_"):
            raise IntentError("invalid draft id")
        with self.lock:
            value = self.values.get(draft_id)
            if (value is not None and source_conversation_id is not None
                    and value.get("source_conversation_id") != source_conversation_id):
                raise IntentError("task draft belongs to another session")
            if value is not None:
                self._require_removable(value)
            return self.values.pop(draft_id, None) is not None

    def invalidate_session(self, session_id: str) -> None:
        with self.lock:
            matching = [key for key, value in self.values.items()
                        if value.get("source_conversation_id") == session_id]
            for key in matching:
                self._require_removable(self.values[key])
            for key in matching:
                self.values.pop(key, None)

    def _prune(self, now: float) -> None:
        for key in [key for key, value in self.values.items()
                    if value["expires_at"] <= now and not value.get("_confirming")
                    and not value.get("_uncertain")]:
            self.values.pop(key, None)

    @staticmethod
    def _require_removable(value: dict) -> None:
        if value.get("_uncertain"):
            raise IntentError(_UNCERTAIN_CREATION)
        if value.get("_confirming"):
            raise IntentError("task draft confirmation already in progress")

    @staticmethod
    def public(value: dict) -> dict:
        result = deepcopy(value)
        result.pop("utterance", None)
        result.pop("_confirming", None)
        result.pop("_decision", None)
        if result.pop("_uncertain", False):
            result["confirmation_state"] = "uncertain"
        return result
