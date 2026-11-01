"""Application-facing logical hardware profile facade."""

from __future__ import annotations

from .base import RuntimeFacade, SDKError


class DeviceError(SDKError):
    def __init__(self, code, message, details=None):
        super().__init__(code, message, details, domain="device")


class DeviceClient(RuntimeFacade):
    """Discover and resolve logical roles without consuming device endpoints."""

    domain = "device"
    error_type = DeviceError

    def profile(self) -> dict:
        return self._call("device.profile", {}, timeout_ms=1_000)

    def inventory(self) -> dict:
        return self._call("device.inventory", {}, timeout_ms=6_000)

    def status(self, *, timeout_ms=6_000) -> dict:
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or not 1 <= timeout_ms <= 6_000:
            raise ValueError("timeout_ms must be between 1 and 6000")
        return self._call("device.status", {}, timeout_ms=timeout_ms)

    def resolve(self, role: str) -> dict:
        if not isinstance(role, str) or not role.strip():
            raise ValueError("role must be a non-empty string")
        return self._call("device.resolve", {"role": role}, timeout_ms=6_000)
