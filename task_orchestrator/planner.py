"""Bounded local-LLM planner with deterministic capability and schedule validation."""

from __future__ import annotations

import json
import re
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


MAX_STEPS = 5
MAX_SKILL_PIPELINE_STEPS = 3
MAX_DELAY_SECONDS = 7 * 24 * 60 * 60
MAX_FUTURE_SECONDS = 366 * 24 * 60 * 60
SAFE_CAPABILITIES = frozenset({
    "system.status", "system.info", "device.status",
    "matrix.status", "matrix.icon", "matrix.blink", "matrix.clear",
    "led.list", "led.set_rgb", "led.off",
    "perception.posture.status",
})
COMMAND_HINTS = {
    "matrix.icon": ("矩阵", "灯", "图标", "爱心", "笑脸", "打勾", "显示"),
    "matrix.blink": ("矩阵", "灯", "闪", "爱心", "笑脸", "提醒"),
    "matrix.clear": ("矩阵", "灯", "清除", "熄灭", "关闭"),
    "led.set_rgb": ("LED", "led", "灯", "颜色", "亮度", "红", "绿", "蓝"),
    "led.off": ("LED", "led", "灯", "熄灭", "关闭"),
}


class PlanError(ValueError):
    pass


class CapabilityCatalog:
    def __init__(self, root: Path | None = None, *, allowed=SAFE_CAPABILITIES):
        self.root = root or Path(__file__).parents[1] / "capability"
        self.allowed = frozenset(allowed)
        self.entries = self._load()

    def _load(self) -> dict:
        entries = {}
        for path in sorted(self.root.glob("*/metadata.json")):
            try:
                values = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise PlanError(f"invalid capability metadata: {path.name}") from exc
            for item in values.get("capabilities", []):
                name = item.get("name")
                if name in self.allowed and item.get("status") == "active":
                    entries[name] = deepcopy(item)
        missing = self.allowed - entries.keys()
        if missing:
            raise PlanError("missing safe capabilities: " + ",".join(sorted(missing)))
        return entries

    def prompt_catalog(self) -> list[dict]:
        return [{
            "name": name,
            "type": item.get("type"),
            "description": item.get("description"),
            "arguments": item.get("input") or {"type": "object"},
        } for name, item in sorted(self.entries.items())]

    def validate_step(self, step: dict, utterance: str) -> dict:
        if not isinstance(step, dict) or set(step) - {"capability", "arguments", "label"}:
            raise PlanError("invalid step fields")
        name, arguments = step.get("capability"), step.get("arguments", {})
        if name not in self.entries or not isinstance(arguments, dict):
            raise PlanError("capability is not allowed")
        label = step.get("label", name)
        if not isinstance(label, str) or not label.strip() or len(label) > 120:
            raise PlanError("invalid step label")
        _validate_schema(self.entries[name].get("input"), arguments, f"{name}.input")
        hints = COMMAND_HINTS.get(name)
        if hints and not any(hint in utterance for hint in hints):
            raise PlanError(f"command not grounded in utterance: {name}")
        return {"capability": name, "arguments": deepcopy(arguments), "label": label.strip()}

    def timeout_ms(self, capability: str) -> int:
        # Runtime's public request contract caps caller-supplied budgets at 30s,
        # even when a legacy capability metadata file advertises a larger value.
        return min(30_000, max(250, int(self.entries[capability].get("timeout_ms", 10_000))))


class TaskPlanner:
    def __init__(self, model, catalog: CapabilityCatalog | None = None, *, skills=None,
                 clock=time.time):
        self.model = model
        self.catalog = catalog or CapabilityCatalog()
        if skills is None:
            from .skills import SkillCatalog
            skills = SkillCatalog()
        self.skills = skills
        self.clock = clock

    def plan(self, utterance: str, *, timezone="UTC") -> dict:
        if not isinstance(utterance, str) or not utterance.strip() or len(utterance) > 1000:
            raise PlanError("utterance must be 1..1000 characters")
        try:
            zone = ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, TypeError, ValueError) as exc:
            raise PlanError("invalid timezone") from exc
        now = datetime.fromtimestamp(self.clock(), zone)
        prompt = (
            "/no_think 你是VENTUNO Q板端任务规划器，只把用户明确要求转换成JSON计划，不执行任务。"
            "只输出一个JSON对象，禁止Markdown、解释和额外字段。格式："
            '{"summary":"简短摘要","schedule":{"kind":"now|delay|at",'
            '"delay_seconds":整数或"run_at":"带时区ISO8601"},'
            '"notification":"任务接收后可朗读的简短中文","steps":['
            '{"capability":"能力名","arguments":{},"label":"步骤说明"}或'
            '{"skill":"技能名","arguments":{},"label":"步骤说明"}]}。'
            "now不带其他时间字段；delay只带delay_seconds；at只带run_at。"
            f"最多{MAX_STEPS}步；纯提醒允许steps为空；包含Skill时最多{MAX_SKILL_PIPELINE_STEPS}步，"
            "只能有一个Skill且必须为最后一步；前置步骤只能使用已注册的安全原子能力。"
            "不得添加用户没有要求的设备副作用。"
            "只有下列能力可用："
            f"{json.dumps(self.catalog.prompt_catalog(), ensure_ascii=False, separators=(',', ':'))}。"
            "只有下列技能可用："
            f"{json.dumps(self.skills.prompt_catalog(), ensure_ascii=False, separators=(',', ':'))}。"
            f"当前本地时间：{now.isoformat()}；时区：{timezone}。"
            f"用户原话：{utterance.strip()}"
        )
        raw = self.model.chat(prompt, max_completion_tokens=512, temperature=0.1)
        now_value = self.clock()
        try:
            return self.validate(raw, utterance=utterance.strip(), now=now_value)
        except PlanError as first_error:
            repair = (
                "/no_think 修复下面的任务计划。只输出修复后的完整JSON对象，不加Markdown或解释；"
                "不得改变用户意图、增加能力或放宽参数。"
                f"验证错误：{first_error}。用户原话：{utterance.strip()}。"
                f"原计划：{str(raw)[:6000]}"
            )
            corrected = self.model.chat(repair, max_completion_tokens=512, temperature=0)
            try:
                return self.validate(corrected, utterance=utterance.strip(), now=now_value)
            except PlanError as second_error:
                raise PlanError(f"plan repair failed: {second_error}") from second_error

    def validate(self, raw, *, utterance: str, now: float | None = None) -> dict:
        value = _parse_json_object(raw) if isinstance(raw, str) else deepcopy(raw)
        if not isinstance(value, dict) or set(value) - {"summary", "schedule", "notification", "steps"}:
            raise PlanError("invalid plan fields")
        summary, notification = value.get("summary"), value.get("notification")
        if not isinstance(summary, str) or not summary.strip() or len(summary) > 200:
            raise PlanError("invalid summary")
        if not isinstance(notification, str) or not notification.strip() or len(notification) > 240:
            raise PlanError("invalid notification")
        steps = value.get("steps")
        if not isinstance(steps, list) or len(steps) > MAX_STEPS:
            raise PlanError("invalid step count")
        normalized_steps = []
        for step in steps:
            if isinstance(step, dict) and "skill" in step:
                normalized_steps.append(self.skills.validate_step(step, utterance))
            else:
                normalized_steps.append(self.catalog.validate_step(step, utterance))
        skill_indexes = [index for index, step in enumerate(normalized_steps) if "skill" in step]
        if skill_indexes and (len(skill_indexes) != 1 or
                              skill_indexes[0] != len(normalized_steps) - 1 or
                              len(normalized_steps) > MAX_SKILL_PIPELINE_STEPS):
            raise PlanError("skill pipeline requires one final skill and at most two safe prelude steps")
        schedule = _validate_schedule(value.get("schedule"), self.clock() if now is None else now)
        return {
            "summary": summary.strip(), "schedule": schedule,
            "notification": notification.strip(), "steps": normalized_steps,
        }


def _parse_json_object(raw: str) -> dict:
    value = raw.strip()
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise PlanError("model did not return JSON")
    try:
        parsed = json.loads(value[start:end + 1])
    except json.JSONDecodeError as exc:
        raise PlanError("model returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise PlanError("plan must be an object")
    return parsed


def _validate_schedule(value, now: float) -> dict:
    if not isinstance(value, dict):
        raise PlanError("schedule must be an object")
    kind = value.get("kind")
    allowed = {
        "now": {"kind"}, "delay": {"kind", "delay_seconds"},
        "at": {"kind", "run_at"},
    }
    if kind not in allowed or set(value) != allowed[kind]:
        raise PlanError("invalid schedule fields")
    if kind == "now":
        run_at = now
    elif kind == "delay":
        delay = value.get("delay_seconds")
        if isinstance(delay, bool) or not isinstance(delay, int) or not 1 <= delay <= MAX_DELAY_SECONDS:
            raise PlanError("delay_seconds out of range")
        run_at = now + delay
    else:
        text = value.get("run_at")
        if not isinstance(text, str) or len(text) > 64:
            raise PlanError("invalid run_at")
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise PlanError("run_at must be ISO8601") from exc
        if parsed.tzinfo is None:
            raise PlanError("run_at must include timezone")
        run_at = parsed.timestamp()
        if run_at < now - 5:
            raise PlanError("run_at is in the past")
        if run_at - now > MAX_FUTURE_SECONDS:
            raise PlanError("run_at is too far in the future")
    return {"kind": kind, "run_at": float(run_at)}


def _validate_schema(schema, value, path):
    if not schema:
        return
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            raise PlanError(f"{path} must be object")
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise PlanError(f"{path} missing {key}")
        if schema.get("additionalProperties") is False and set(value) - properties.keys():
            raise PlanError(f"{path} has unexpected arguments")
        for key, item in value.items():
            if key in properties:
                _validate_schema(properties[key], item, f"{path}.{key}")
    elif kind == "array":
        if not isinstance(value, list):
            raise PlanError(f"{path} must be array")
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", 1_000_000):
            raise PlanError(f"{path} array length invalid")
        for index, item in enumerate(value):
            _validate_schema(schema.get("items"), item, f"{path}[{index}]")
    elif kind == "string":
        if not isinstance(value, str):
            raise PlanError(f"{path} must be string")
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            raise PlanError(f"{path} pattern mismatch")
    elif kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise PlanError(f"{path} must be integer")
    elif kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PlanError(f"{path} must be number")
    elif kind == "boolean" and not isinstance(value, bool):
        raise PlanError(f"{path} must be boolean")
    if "enum" in schema and value not in schema["enum"]:
        raise PlanError(f"{path} enum mismatch")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema.get("minimum", value) or value > schema.get("maximum", value):
            raise PlanError(f"{path} out of range")
