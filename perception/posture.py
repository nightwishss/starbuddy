"""Deterministic temporal policy for local ergonomic posture observations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PostureSignal:
    """Semantic pose output; raw frames and landmarks stay below this boundary."""

    observed_at: float
    visible: bool
    confidence: float
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.observed_at < 0:
            raise ValueError("observed_at must be non-negative")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        allowed = {"head_forward", "torso_lean", "shoulder_asymmetry", "too_close"}
        if any(reason not in allowed for reason in self.reasons):
            raise ValueError("unsupported posture reason")


@dataclass(frozen=True)
class PostureDecision:
    state: str
    reasons: tuple[str, ...]
    stable_for_seconds: float
    remind: bool


class PostureTemporalPolicy:
    """Confirm persistent deviations and rate-limit ergonomic reminders."""

    def __init__(self, *, confirmation_seconds=20.0, reminder_cooldown_seconds=180.0,
                 recovery_seconds=5.0, minimum_confidence=0.6,
                 missing_grace_seconds=1.0, auto_mark_reminder=True) -> None:
        if confirmation_seconds <= 0 or reminder_cooldown_seconds <= 0 or recovery_seconds <= 0:
            raise ValueError("time bounds must be positive")
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if missing_grace_seconds < 0:
            raise ValueError("missing_grace_seconds must be non-negative")
        if type(auto_mark_reminder) is not bool:
            raise ValueError("auto_mark_reminder must be boolean")
        self.confirmation_seconds = float(confirmation_seconds)
        self.reminder_cooldown_seconds = float(reminder_cooldown_seconds)
        self.recovery_seconds = float(recovery_seconds)
        self.minimum_confidence = float(minimum_confidence)
        self.missing_grace_seconds = float(missing_grace_seconds)
        self.auto_mark_reminder = auto_mark_reminder
        self.bad_since: float | None = None
        self.bad_observed_seconds = 0.0
        self.last_reliable_bad_at: float | None = None
        self.last_reliable_at: float | None = None
        self.missing_since: float | None = None
        self.recovery_since: float | None = None
        self.last_reminder_at: float | None = None
        self.last_observed_at: float | None = None
        self.active_reasons: tuple[str, ...] = ()

    def _clear_candidate(self) -> None:
        self.bad_since = None
        self.bad_observed_seconds = 0.0
        self.last_reliable_bad_at = None
        self.recovery_since = None
        self.active_reasons = ()

    def update(self, signal: PostureSignal) -> PostureDecision:
        if self.last_observed_at is not None and signal.observed_at < self.last_observed_at:
            raise ValueError("observations must be chronological")
        self.last_observed_at = signal.observed_at
        if not signal.visible or signal.confidence < self.minimum_confidence:
            if self.missing_since is None:
                self.missing_since = signal.observed_at
            if (self.last_reliable_at is None or
                    signal.observed_at - self.last_reliable_at > self.missing_grace_seconds):
                self._clear_candidate()
            # Neither missing intervals nor recovery intervals are bad-pose
            # evidence. Always remain unknown and silent while missing.
            self.last_reliable_bad_at = None
            self.recovery_since = None
            return PostureDecision("unknown", (), 0.0, False)

        if (self.missing_since is not None and
                (self.last_reliable_at is None or
                 signal.observed_at - self.last_reliable_at > self.missing_grace_seconds)):
            self._clear_candidate()
        self.missing_since = None
        self.last_reliable_at = signal.observed_at

        reasons = tuple(sorted(set(signal.reasons)))
        if reasons:
            self.recovery_since = None
            if self.bad_since is None:
                self.bad_since = signal.observed_at
            if self.last_reliable_bad_at is not None:
                self.bad_observed_seconds += signal.observed_at - self.last_reliable_bad_at
            self.last_reliable_bad_at = signal.observed_at
            self.active_reasons = reasons
            stable = self.bad_observed_seconds
            if stable < self.confirmation_seconds:
                return PostureDecision("suspect", reasons, stable, False)
            remind = (
                self.last_reminder_at is None
                or signal.observed_at - self.last_reminder_at >= self.reminder_cooldown_seconds
            )
            if remind and self.auto_mark_reminder:
                self.last_reminder_at = signal.observed_at
            return PostureDecision("unhealthy", reasons, stable, remind)

        self.last_reliable_bad_at = None
        if self.bad_since is None:
            return PostureDecision("healthy", (), 0.0, False)
        if self.recovery_since is None:
            self.recovery_since = signal.observed_at
        recovered_for = signal.observed_at - self.recovery_since
        if recovered_for < self.recovery_seconds:
            return PostureDecision("suspect", self.active_reasons, self.bad_observed_seconds, False)
        self._clear_candidate()
        return PostureDecision("healthy", (), recovered_for, False)
