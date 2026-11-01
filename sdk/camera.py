"""Application-facing bounded still-camera facade."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from .base import RuntimeFacade, SDKError


class CameraError(SDKError):
    def __init__(self, code, message, details=None):
        super().__init__(code, message, details, domain="camera")


class CameraClient(RuntimeFacade):
    """Use stable IDs or logical roles; never expose pixels or capture paths."""

    domain = "camera"
    error_type = CameraError

    def list(self) -> dict[str, Any]:
        return self._call("camera.list", {}, timeout_ms=5_000)

    def get_info(self, camera_id: str) -> dict[str, Any]:
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ValueError("camera_id must be a non-empty string")
        return self._call("camera.get_info", {"camera_id": camera_id}, timeout_ms=5_000)

    def snapshot(self, *, camera_id: str | None = None, role: str | None = None,
                 source: dict[str, Any] | None = None,
                 transform: dict[str, Any] | None = None,
                 quality: int | None = None,
                 request_id: str | None = None) -> dict[str, Any]:
        if bool(camera_id) == bool(role):
            raise ValueError("provide exactly one of camera_id or role")
        arguments: dict[str, Any] = {"camera_id": camera_id} if camera_id else {"role": role}
        if source is not None:
            arguments["source"] = dict(source)
        if transform is not None:
            arguments["transform"] = dict(transform)
        if quality is not None:
            arguments["quality"] = quality
        return self._call(
            "camera.snapshot", arguments, timeout_ms=15_000, request_id=request_id
        )

    def frame_info(self, frame_id: str) -> dict[str, Any]:
        return self._call("camera.frame_info", {"frame_id": frame_id}, timeout_ms=1_000)

    def release(self, frame_id: str) -> dict[str, Any]:
        return self._call("camera.release_frame", {"frame_id": frame_id}, timeout_ms=1_000)

    @contextmanager
    def frame(self, **snapshot_arguments) -> Iterator[dict[str, Any]]:
        """Capture one frame and guarantee release on normal or exceptional exit."""
        value = self.snapshot(**snapshot_arguments)
        try:
            yield value
        finally:
            self.release(value["frame_id"])
