"""Validate bounded conversational proposals without performing their effects.

The caller supplies the current session and turn liveness immediately before
dispatch. Text validation is structural; perception facts still require the
existing evidence checks in the conversation pipeline.
"""

from __future__ import annotations

import math
import re
import time
from copy import deepcopy

from interaction_context.intent import explicit_posture_snooze_seconds
from task_orchestrator.planner import TaskPlanner


MAX_PROPOSAL_TTL_SECONDS = 600
_KINDS = frozenset({"reply", "clarify", "wait", "task_draft", "control"})
_ROUTES = frozenset({"rule", "local", "cloud"})
_COMMON_FIELDS = frozenset({
    "schema_version", "kind", "route", "reason", "session_id", "created_at",
    "expires_at", "context_version", "text", "model_calls", "utterance",
})
_REQUIRED_FIELDS = frozenset({
    "schema_version", "kind", "route", "reason", "session_id", "created_at",
    "expires_at", "text",
})
_STOP_UTTERANCES = frozenset({
    "停止坐姿守护", "暂停坐姿守护", "停止坐姿监测", "不用看坐姿了",
})


class DecisionError(ValueError):
    """A proposal cannot be admitted in the current turn."""


class DecisionGate:
    def __init__(self, planner=None, clock=time.time):
        self.clock = clock
        self.planner = planner if planner is not None else TaskPlanner(None, clock=clock)

    def validate(self, proposal, *, session_id, is_active=lambda: True) -> dict:
        """Return an isolated, validated proposal, retaining its raw schedule.

        A validated proposal grants no durable permission: callers must validate
        again at a later confirmation or dispatch boundary.
        """
        self._require_active(is_active)
        if not isinstance(proposal, dict):
            raise DecisionError("decision must be an object")
        value = deepcopy(proposal)
        kind = value.get("kind")
        if not isinstance(kind, str) or kind not in _KINDS:
            raise DecisionError("invalid decision kind")
        extra = {"plan"} if kind == "task_draft" else {"control"} if kind == "control" else set()
        if set(value) - (_COMMON_FIELDS | extra) or _REQUIRED_FIELDS - set(value):
            raise DecisionError("invalid decision fields")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise DecisionError("invalid decision schema version")
        route = value["route"]
        if not isinstance(route, str) or route not in _ROUTES:
            raise DecisionError("invalid decision route")
        reason = value["reason"]
        if not isinstance(reason, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", reason):
            raise DecisionError("invalid decision reason")
        if (not isinstance(session_id, str)
                or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}", session_id)
                or value["session_id"] != session_id):
            raise DecisionError("decision belongs to another session")
        if value.get("context_version") is not None and not isinstance(value["context_version"], str):
            raise DecisionError("invalid context version")
        if "model_calls" in value and (
                type(value["model_calls"]) is not int or not 0 <= value["model_calls"] <= 2):
            raise DecisionError("invalid model call count")
        text = value["text"]
        text_limit = 1200 if kind == "reply" else 240
        if (not isinstance(text, str) or len(text) > text_limit
                or (kind in {"reply", "clarify", "wait"} and not text.strip())):
            raise DecisionError("invalid decision text")
        if "utterance" in value or kind in {"task_draft", "control"}:
            utterance = value.get("utterance")
            if not isinstance(utterance, str) or not utterance.strip() or len(utterance) > 1000:
                raise DecisionError("utterance must be 1..1000 characters")
        self._require_fresh(value)
        if kind == "task_draft":
            if not isinstance(value.get("plan"), dict):
                raise DecisionError("task draft requires a plan")
            try:
                # Task creation performs the same validation. Keep the original
                # schedule: its normalized run_at is not a valid raw delay/now.
                self.planner.validate(
                    value["plan"], utterance=value["utterance"], now=self.clock(),
                )
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                raise DecisionError("invalid task plan") from exc
        elif kind == "control":
            self._validate_control(value)
        self._require_active(is_active)
        self._require_fresh(value)
        return value

    @staticmethod
    def _require_active(is_active):
        try:
            active = is_active()
        except Exception as exc:
            raise DecisionError("decision turn unavailable") from exc
        if active is not True:
            raise DecisionError("decision turn is cancelled")

    def _require_fresh(self, value):
        created, expires, now = value["created_at"], value["expires_at"], self.clock()
        try:
            finite = all(type(item) in (int, float) and math.isfinite(item)
                         for item in (created, expires, now))
        except (OverflowError, TypeError, ValueError):
            finite = False
        if not finite:
            raise DecisionError("invalid decision timestamps")
        if not 0 < expires - created <= MAX_PROPOSAL_TTL_SECONDS:
            raise DecisionError("invalid decision lifetime")
        if created > now or expires <= now:
            raise DecisionError("decision is expired or not yet current")

    @staticmethod
    def _validate_control(value):
        if value["route"] != "rule":
            raise DecisionError("control must come from an explicit rule")
        control = value.get("control")
        if not isinstance(control, dict):
            raise DecisionError("control decision requires a control")
        name = control.get("name")
        if name == "posture_stop":
            if set(control) != {"name"} or value["utterance"].strip(" 。！？!?,，") not in _STOP_UTTERANCES:
                raise DecisionError("posture stop is not explicit")
        elif name == "posture_snooze":
            if set(control) != {"name", "seconds"} or type(control["seconds"]) is not int:
                raise DecisionError("invalid posture snooze fields")
            try:
                seconds = explicit_posture_snooze_seconds(value["utterance"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise DecisionError("invalid posture snooze request") from exc
            if seconds is None or control["seconds"] != seconds:
                raise DecisionError("posture snooze is not explicit")
        else:
            raise DecisionError("control is not allowed")
