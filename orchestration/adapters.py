"""Replaceable expression outputs with per-adapter failure isolation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .interaction import ExpressionIntent, InteractionPhase


LOG = logging.getLogger("vq.expression")


@dataclass(frozen=True)
class AdapterResult:
    adapter: str
    ok: bool
    detail: str

    def public(self) -> dict:
        return {"adapter": self.adapter, "ok": self.ok, "detail": self.detail}


class CompositeExpressionAdapter:
    """Fan out an intent without allowing one device to break the interaction."""

    def __init__(self, adapters=()):
        self.adapters = tuple(adapters)

    def apply(self, intent: ExpressionIntent, trace_id: str) -> list[AdapterResult]:
        results = []
        for adapter in self.adapters:
            name = getattr(adapter, "name", type(adapter).__name__)
            try:
                detail = adapter.apply(intent, trace_id)
                results.append(AdapterResult(name, True, str(detail or "applied")))
            except Exception as exc:  # device adapters are an explicit failure boundary
                LOG.warning("expression adapter failed adapter=%s trace_id=%s error=%s", name, trace_id, type(exc).__name__)
                results.append(AdapterResult(name, False, "device unavailable"))
        return results


class MatrixExpressionAdapter:
    """Map abstract phases to the current 8x13 monochrome development matrix."""

    name = "matrix-8x13"
    _THINKING = [
        int(bit) * 3
        for row in (
            "0000000000000", "0000000000000", "0000000000000", "0010001000100",
            "0010001000100", "0000000000000", "0000000000000", "0000000000000",
        )
        for bit in row
    ]

    def __init__(self, matrix):
        self.matrix = matrix

    def apply(self, intent: ExpressionIntent, trace_id: str) -> str:
        if intent.phase == InteractionPhase.IDLE:
            self.matrix.icon("smile", 2, request_id=trace_id)
        elif intent.phase == InteractionPhase.LISTENING:
            self.matrix.icon("heart", 4, request_id=trace_id)
        elif intent.phase == InteractionPhase.THINKING:
            self.matrix.draw(self._THINKING, request_id=trace_id)
        elif intent.phase == InteractionPhase.SPEAKING:
            self.matrix.icon("smile", 5, request_id=trace_id)
        else:
            self.matrix.icon("cross", 3, request_id=trace_id)
        return intent.phase.value


class DisplayExpressionAdapter:
    """Map hardware-neutral expression intent to the loopback HDMI surface."""

    name = "hdmi-product-display"
    _COPY = {
        InteractionPhase.IDLE: ("你好，我是星伴", "我在这里陪着你"),
        InteractionPhase.LISTENING: ("我在听", "自然说话就好，说完停一下"),
        InteractionPhase.THINKING: ("让我想一想", "正在板端理解和安排"),
        InteractionPhase.SPEAKING: ("我来回答你", "声音正在从板载扬声器播放"),
        InteractionPhase.ERROR: ("刚才没有听清", "靠近一点，我们再试一次"),
    }

    def __init__(self, display):
        self.display = display

    def apply(self, intent: ExpressionIntent, trace_id: str) -> str:
        primary, secondary = self._COPY[intent.phase]
        self.display.update(
            phase=intent.phase.value,
            visual_style=intent.visual_style,
            primary_text=primary,
            secondary_text=secondary,
            trace_id=trace_id,
        )
        return intent.phase.value
