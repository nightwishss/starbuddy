"""Bounded reusable behaviors compiled to registered Runtime capabilities."""

from __future__ import annotations

from copy import deepcopy

from .planner import PlanError, _validate_schema


SKILLS = {
    "monitor_posture": {
        "description": "在限定时长内运行个人坐姿守护；持续判断和提醒由姿态策略负责。",
        "capability": "perception.posture.monitor",
        "stop_capability": "perception.posture.stop",
        "input": {
            "type": "object",
            "properties": {
                "duration_seconds": {"type": "integer", "minimum": 60, "maximum": 14_400},
                "bad_duration_seconds": {"type": "integer", "minimum": 5, "maximum": 120},
                "cooldown_seconds": {"type": "integer", "minimum": 30, "maximum": 1_800},
            },
            "required": ["duration_seconds"],
            "additionalProperties": False,
        },
        "defaults": {"bad_duration_seconds": 20, "cooldown_seconds": 180},
        "hints": ("坐姿", "姿势", "驼背", "体态"),
        # Runtime currently serializes some hardware/model effects. A task may
        # be confirmed while the confirmation TTS is still finishing, so use
        # Runtime's public maximum while the provider keeps its own 3s HTTP bound.
        "timeout_ms": 30_000,
    },
}


class SkillCatalog:
    def __init__(self, definitions=None):
        self.entries = deepcopy(definitions or SKILLS)

    def prompt_catalog(self) -> list[dict]:
        return [{
            "name": name, "description": value["description"],
            "arguments": value["input"],
        } for name, value in sorted(self.entries.items())]

    def validate_step(self, step: dict, utterance: str) -> dict:
        if not isinstance(step, dict) or set(step) - {"skill", "arguments", "label"}:
            raise PlanError("invalid skill step fields")
        name, arguments = step.get("skill"), step.get("arguments", {})
        if name not in self.entries or not isinstance(arguments, dict):
            raise PlanError("skill is not allowed")
        entry = self.entries[name]
        normalized = {**entry.get("defaults", {}), **deepcopy(arguments)}
        _validate_schema(entry["input"], normalized, f"{name}.input")
        if not any(hint in utterance for hint in entry.get("hints", ())):
            raise PlanError(f"skill not grounded in utterance: {name}")
        label = step.get("label", name)
        if not isinstance(label, str) or not label.strip() or len(label) > 120:
            raise PlanError("invalid skill step label")
        return {"skill": name, "arguments": normalized, "label": label.strip()}

    def implementation(self, name: str) -> dict:
        try:
            return deepcopy(self.entries[name])
        except KeyError as exc:
            raise PlanError("skill is not allowed") from exc


class RuntimeSkillExecutor:
    """Execute admission of a bounded behavior; the provider owns its temporal loop."""

    def __init__(self, runtime, catalog: SkillCatalog | None = None):
        self.runtime = runtime
        self.catalog = catalog or SkillCatalog()

    def call(self, name: str, arguments: dict, *, task_id: str, request_id: str):
        entry = self.catalog.implementation(name)
        values = {**deepcopy(arguments), "task_id": task_id}
        return self.runtime.call(
            entry["capability"], values, timeout_ms=entry["timeout_ms"], request_id=request_id,
        )

    def stop(self, name: str, *, task_id: str, request_id: str):
        entry = self.catalog.implementation(name)
        capability = entry.get("stop_capability")
        if not capability:
            raise PlanError("skill does not support stop")
        return self.runtime.call(
            capability, {"task_id": task_id}, timeout_ms=3_000, request_id=request_id,
        )
