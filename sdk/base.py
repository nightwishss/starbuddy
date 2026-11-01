"""Shared application-SDK error and Runtime forwarding boundary."""

from __future__ import annotations

from typing import Any

from runtime.client import RuntimeClient


class SDKError(RuntimeError):
    """A typed Runtime capability failure safe for application handling."""

    def __init__(self, code: str, message: str,
                 details: dict[str, Any] | None = None, *, domain: str = "platform"):
        super().__init__(f"{code}: {message}")
        self.domain = domain
        self.code = code
        self.details = details or {}


class RuntimeFacade:
    """Route a domain SDK exclusively through the reviewed Runtime contract."""

    domain = "platform"
    error_type = SDKError

    def __init__(self, runtime: RuntimeClient | None = None):
        self._runtime = runtime or RuntimeClient()

    def _call(self, capability: str, arguments: dict[str, Any], *,
              timeout_ms: int | None = None,
              request_id: str | None = None) -> dict[str, Any]:
        options = {}
        if timeout_ms is not None:
            options["timeout_ms"] = timeout_ms
        if request_id is not None:
            options["request_id"] = request_id
        try:
            response = self._runtime.call(capability, arguments, **options)
        except ValueError as exc:
            raise self.error_type(
                "REQUEST_ERROR", f"{self.domain} capability request was rejected"
            ) from exc
        except (OSError, RuntimeError) as exc:
            raise self.error_type(
                "TRANSPORT_ERROR", f"{self.domain} Runtime transport is unavailable"
            ) from exc
        if not isinstance(response, dict):
            raise self.error_type(
                "PROTOCOL_ERROR", f"{self.domain} capability returned an invalid response"
            )
        if not response.get("ok"):
            error = response.get("error") or {}
            raise self.error_type(
                str(error.get("code", "RUNTIME_ERROR")),
                str(error.get("message", f"{self.domain} capability failed")),
                error.get("details") if isinstance(error.get("details"), dict) else None,
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise self.error_type(
                "PROTOCOL_ERROR", f"{self.domain} capability returned an invalid result"
            )
        return result
