"""Conservative, explainable affect candidate estimation v0."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AffectFeatures:
    text: str = ""
    rms: float | None = None
    speech_rate: float | None = None
    pause_ratio: float | None = None

    def __post_init__(self):
        for name in ("rms", "speech_rate", "pause_ratio"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class AffectEstimate:
    valence: float
    arousal: float
    confidence: float
    label: str
    evidence: tuple[str, ...]


class AffectEstimator:
    _NEGATIVE = ("难过", "伤心", "害怕", "孤单", "不开心", "紧张")
    _POSITIVE = ("开心", "高兴", "喜欢", "太好了", "兴奋")

    def estimate(self, features: AffectFeatures) -> AffectEstimate:
        evidence = []
        valence = 0.0
        arousal = 0.0
        for word in self._NEGATIVE:
            if word in features.text:
                valence -= 0.45; evidence.append("explicit_negative_text"); break
        for word in self._POSITIVE:
            if word in features.text:
                valence += 0.45; evidence.append("explicit_positive_text"); break
        if features.rms is not None:
            if features.rms > 0.75: arousal += 0.3; evidence.append("high_energy")
            elif features.rms < 0.2: arousal -= 0.2; evidence.append("low_energy")
        if features.speech_rate is not None and features.speech_rate > 0.75:
            arousal += 0.25; evidence.append("fast_speech")
        if features.pause_ratio is not None and features.pause_ratio > 0.6:
            arousal -= 0.2; evidence.append("long_pauses")
        confidence = min(0.9, 0.35 + 0.2 * len(set(evidence))) if evidence else 0.0
        label = "neutral"
        if valence <= -0.25: label = "gentle_support"
        elif valence >= 0.25 and arousal >= 0.2: label = "celebrate"
        elif valence >= 0.25: label = "positive"
        return AffectEstimate(max(-1.0, valence), max(-1.0, min(1.0, arousal)), confidence, label, tuple(evidence))
