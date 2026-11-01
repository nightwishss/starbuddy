"""Application-facing owner-bound local speech facade."""

from __future__ import annotations

from typing import Any

from .base import RuntimeFacade, SDKError


class SpeechError(SDKError):
    def __init__(self, code, message, details=None):
        super().__init__(code, message, details, domain="speech")


class SpeechClient(RuntimeFacade):
    """Run local STT/TTS only against explicitly owned shared-media objects."""

    domain = "speech"
    error_type = SpeechError

    def status(self) -> dict[str, Any]:
        return self._call("speech.status", {}, timeout_ms=1_000)

    def transcribe(self, media_id: str, media_owner: str) -> dict[str, Any]:
        if not media_id or not media_owner:
            raise ValueError("media_id and media_owner are required")
        return self._call("speech.transcribe", {
            "media_id": media_id, "media_owner": media_owner,
        }, timeout_ms=30_000)

    def synthesize(self, text: str, media_owner: str, *, speed=1.0) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be non-empty")
        if not media_owner:
            raise ValueError("media_owner is required")
        if type(speed) not in (int, float) or not 0.6 <= speed <= 1.0:
            raise ValueError("speed must be 0.6..1.0")
        return self._call("speech.synthesize", {
            "text": text, "media_owner": media_owner,
            **({"speed": speed} if speed != 1.0 else {}),
        }, timeout_ms=30_000)
