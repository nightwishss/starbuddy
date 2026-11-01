"""Application-facing interface for the VENTUNO Q 8x13 LED matrix.

This module deliberately talks only to VQ Runtime. Applications must not
depend on the relay socket, Arduino App container, RouterBridge, or MCU sketch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import RuntimeFacade, SDKError


class MatrixError(SDKError):
    """A matrix capability call was rejected or could not be completed."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(code, message, details, domain="matrix")


class MatrixClient(RuntimeFacade):
    """Stable synchronous facade over the versioned ``matrix.*`` contracts."""

    ROWS = 8
    COLUMNS = 13
    PIXELS = ROWS * COLUMNS
    ICONS = frozenset({"check", "cross", "heart", "smile"})

    domain = "matrix"
    error_type = MatrixError

    def status(self) -> dict[str, Any]:
        return self._call("matrix.status", {})

    def icon(self, name: str, brightness: int = 7, *, request_id: str | None = None) -> dict[str, Any]:
        return self._call(
            "matrix.icon", {"name": name, "brightness": brightness}, request_id=request_id
        )

    def draw(self, pixels: Sequence[int], *, request_id: str | None = None) -> dict[str, Any]:
        return self._call("matrix.draw", {"pixels": list(pixels)}, request_id=request_id)

    def blink(
        self,
        name: str,
        *,
        brightness: int = 7,
        duration_seconds: int = 10,
        interval_ms: int = 500,
        end_state: str = "on",
    ) -> dict[str, Any]:
        return self._call("matrix.blink", {
            "name": name,
            "brightness": brightness,
            "duration_seconds": duration_seconds,
            "interval_ms": interval_ms,
            "end_state": end_state,
        })

    def clear(self) -> dict[str, Any]:
        return self._call("matrix.clear", {})
