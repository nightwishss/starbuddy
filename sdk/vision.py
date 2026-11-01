"""Application-facing facade for local object detection on owned media."""

from __future__ import annotations

from typing import Any

from .base import RuntimeFacade, SDKError


class VisionError(SDKError):
    def __init__(self, code, message, details=None):
        super().__init__(code, message, details, domain="vision")


class VisionClient(RuntimeFacade):
    domain = "vision"
    error_type = VisionError

    def detect_objects(self, media_id: str, media_owner: str, *, confidence=0.4) -> dict[str, Any]:
        if not media_id or not media_owner:
            raise ValueError("media_id and media_owner are required")
        return self._call("vision.detect_objects", {
            "media_id": media_id, "media_owner": media_owner,
            "confidence": confidence,
        }, timeout_ms=15_000)
