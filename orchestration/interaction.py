"""Pure state machine that converts pipeline events into expression intents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class InvalidTransition(RuntimeError):
    pass


class InteractionPhase(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


class InteractionEvent(str, Enum):
    LISTEN_STARTED = "listen_started"
    INPUT_READY = "input_ready"
    RESPONSE_READY = "response_ready"
    SPEECH_FINISHED = "speech_finished"
    FAILED = "failed"
    RESET = "reset"


@dataclass(frozen=True)
class Affect:
    """Bounded, explainable affect estimate; zero is neutral."""

    valence: float = 0.0
    arousal: float = 0.0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between -1 and 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class ExpressionIntent:
    """Hardware-neutral output requested by the interaction core."""

    phase: InteractionPhase
    visual_style: str
    light_style: str
    motion_style: str
    speech_rate: float
    speech_volume: float
    interruptible: bool

    def public(self) -> dict:
        value = asdict(self)
        value["phase"] = self.phase.value
        return value


class InteractionOrchestrator:
    _TRANSITIONS = {
        InteractionPhase.IDLE: {
            InteractionEvent.LISTEN_STARTED: InteractionPhase.LISTENING,
            InteractionEvent.INPUT_READY: InteractionPhase.THINKING,
            InteractionEvent.FAILED: InteractionPhase.ERROR,
            InteractionEvent.RESET: InteractionPhase.IDLE,
        },
        InteractionPhase.LISTENING: {
            InteractionEvent.INPUT_READY: InteractionPhase.THINKING,
            InteractionEvent.FAILED: InteractionPhase.ERROR,
            InteractionEvent.RESET: InteractionPhase.IDLE,
        },
        InteractionPhase.THINKING: {
            InteractionEvent.RESPONSE_READY: InteractionPhase.SPEAKING,
            InteractionEvent.FAILED: InteractionPhase.ERROR,
            InteractionEvent.RESET: InteractionPhase.IDLE,
        },
        InteractionPhase.SPEAKING: {
            InteractionEvent.SPEECH_FINISHED: InteractionPhase.IDLE,
            InteractionEvent.LISTEN_STARTED: InteractionPhase.LISTENING,
            InteractionEvent.FAILED: InteractionPhase.ERROR,
            InteractionEvent.RESET: InteractionPhase.IDLE,
        },
        InteractionPhase.ERROR: {
            InteractionEvent.RESET: InteractionPhase.IDLE,
            InteractionEvent.LISTEN_STARTED: InteractionPhase.LISTENING,
        },
    }

    def __init__(self) -> None:
        self.phase = InteractionPhase.IDLE
        self.affect = Affect()
        self.sequence = 0

    def update_affect(self, affect: Affect) -> ExpressionIntent:
        self.affect = affect
        self.sequence += 1
        return self.intent()

    def transition(self, event: InteractionEvent) -> ExpressionIntent:
        try:
            target = self._TRANSITIONS[self.phase][event]
        except KeyError as exc:
            raise InvalidTransition(f"{self.phase.value} cannot accept {event.value}") from exc
        self.phase = target
        self.sequence += 1
        return self.intent()

    def intent(self) -> ExpressionIntent:
        calm = self.affect.confidence >= 0.5 and self.affect.valence < -0.25
        styles = {
            InteractionPhase.IDLE: ("calm", "ambient", "still", 1.0, 0.72, True),
            InteractionPhase.LISTENING: ("attentive", "listening", "lean_in", 1.0, 0.0, True),
            InteractionPhase.THINKING: ("thinking", "thinking", "breathe", 1.0, 0.0, True),
            InteractionPhase.SPEAKING: (
                "gentle" if calm else "speaking",
                "gentle" if calm else "speaking",
                "soft_nod" if calm else "talk",
                0.9 if calm else 1.0,
                0.62 if calm else 0.78,
                True,
            ),
            InteractionPhase.ERROR: ("concerned", "error_soft", "still", 0.92, 0.6, True),
        }
        visual, light, motion, rate, volume, interruptible = styles[self.phase]
        return ExpressionIntent(self.phase, visual, light, motion, rate, volume, interruptible)

    def snapshot(self) -> dict:
        return {
            "sequence": self.sequence,
            "phase": self.phase.value,
            "affect": asdict(self.affect),
            "intent": self.intent().public(),
        }
