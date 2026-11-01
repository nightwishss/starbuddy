"""Application-facing client for structured, pixel-free observations."""

from __future__ import annotations

from typing import Any

from .base import RuntimeFacade, SDKError


class PerceptionError(SDKError):
    def __init__(self, code, message, details=None):
        super().__init__(code, message, details, domain="perception")


class PerceptionClient(RuntimeFacade):
    domain = "perception"
    error_type = PerceptionError

    def observe(self, *, role="camera.environment", profile="fast", confidence=0.35,
                stability_samples=2) -> dict[str, Any]:
        return self._call("perception.observe", {
            "role": role, "profile": profile, "confidence": confidence,
            "stability_samples": stability_samples,
        }, timeout_ms=30_000)

    def posture_status(self) -> dict[str, Any]:
        return self._call("perception.posture.status", {}, timeout_ms=3_000)

    def posture_start(self) -> dict[str, Any]:
        return self._call("perception.posture.start", {}, timeout_ms=3_000)

    def posture_pause(self) -> dict[str, Any]:
        return self._call("perception.posture.pause", {}, timeout_ms=3_000)

    def posture_calibrate(self) -> dict[str, Any]:
        return self._call("perception.posture.calibrate", {}, timeout_ms=3_000)

    def posture_reset(self) -> dict[str, Any]:
        return self._call("perception.posture.reset", {}, timeout_ms=3_000)

    def posture_simulate(self) -> dict[str, Any]:
        return self._call("perception.posture.simulate", {}, timeout_ms=3_000)
