"""Trusted-host facade over the legacy Runtime display-session contract."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .base import RuntimeFacade, SDKError


class DisplayError(SDKError):
    def __init__(self, code, message, details=None):
        super().__init__(code, message, details, domain="display")


class DisplayClient(RuntimeFacade):
    """Own bounded software display sessions; this is not the HDMI kiosk API."""

    domain = "display"
    error_type = DisplayError

    def list(self) -> dict[str, Any]:
        return self._call("display.list", {}, timeout_ms=1_000)

    def get_info(self) -> dict[str, Any]:
        return self._call("display.get_info", {}, timeout_ms=1_000)

    def open(self, *, width=1280, height=720, backend="framebuffer",
             display: str | None = None, title: str | None = None) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "width": width, "height": height, "backend": backend,
        }
        if display is not None:
            arguments["display"] = display
        if title is not None:
            arguments["title"] = title
        return self._call("display.open", arguments, timeout_ms=2_000)

    def show_text(self, display_id: str, text: str,
                  position: Sequence[int] | None = None) -> dict[str, Any]:
        arguments: dict[str, Any] = {"display_id": display_id, "text": text}
        if position is not None:
            arguments["position"] = list(position)
        return self._call("display.show_text", arguments, timeout_ms=2_000)

    def show_image_path(self, display_id: str, path: str | Path) -> dict[str, Any]:
        """Render a trusted local path; untrusted apps should use shared media instead."""
        return self._call("display.show_image", {
            "display_id": display_id, "path": str(path),
        }, timeout_ms=3_000)

    def clear(self, display_id: str, color: str = "black") -> dict[str, Any]:
        return self._call("display.clear", {
            "display_id": display_id, "color": color,
        }, timeout_ms=2_000)

    def capture_path(self, display_id: str, path: str | Path) -> dict[str, Any]:
        """Capture to a trusted local path pending a shared-media display contract."""
        return self._call("display.capture", {
            "display_id": display_id, "path": str(path),
        }, timeout_ms=3_000)

    def close(self, display_id: str) -> dict[str, Any]:
        return self._call("display.close", {"display_id": display_id}, timeout_ms=2_000)

    @contextmanager
    def session(self, **open_arguments) -> Iterator[dict[str, Any]]:
        value = self.open(**open_arguments)
        try:
            yield value
        finally:
            self.close(value["display_id"])
