"""Deterministic child-safety pre-policy; never delegates escalation to a model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SafetyLevel(str, Enum):
    NORMAL = "normal"
    DISTRESS = "distress"
    DANGER = "danger"
    IMMEDIATE = "immediate"


@dataclass(frozen=True)
class SafetyDecision:
    level: SafetyLevel
    allow_model: bool
    require_trusted_adult: bool
    response: str | None
    evidence: tuple[str, ...]


class ChildSafetyPolicy:
    """Small auditable v0 policy; this is not a clinical classifier."""

    _IMMEDIATE = ("想自杀", "不想活", "杀了自己", "马上伤害自己", "有人正在打我", "现在有危险")
    _DANGER = ("被打", "被欺负", "有人摸我", "不敢回家", "要伤害我", "离家出走")
    _DISTRESS = ("难过", "害怕", "孤单", "一直哭", "没人喜欢我", "不开心")

    def evaluate(self, text: str) -> SafetyDecision:
        normalized = "".join(text.lower().split())
        immediate = tuple(item for item in self._IMMEDIATE if item in normalized)
        if immediate:
            return SafetyDecision(
                SafetyLevel.IMMEDIATE, False, True,
                "你现在的安全最重要。请马上去找身边可信任的大人，并联系当地紧急求助服务；不要一个人待着。",
                immediate,
            )
        danger = tuple(item for item in self._DANGER if item in normalized)
        if danger:
            return SafetyDecision(
                SafetyLevel.DANGER, False, True,
                "这件事需要可信任的大人帮助你。请尽快告诉家长、老师或其他能保护你的成年人。",
                danger,
            )
        distress = tuple(item for item in self._DISTRESS if item in normalized)
        if distress:
            return SafetyDecision(SafetyLevel.DISTRESS, True, False, None, distress)
        return SafetyDecision(SafetyLevel.NORMAL, True, False, None, ())
