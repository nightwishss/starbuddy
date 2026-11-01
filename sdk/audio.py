"""Application-facing physical audio facade over VQ Runtime."""

from __future__ import annotations

from typing import Any

from .base import RuntimeFacade, SDKError


class AudioError(SDKError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(code, message, details, domain="audio")


class AudioClient(RuntimeFacade):
    """Use logical devices and media IDs; never expose PipeWire nodes or paths."""

    domain = "audio"
    error_type = AudioError

    def capture_clip(self, media_owner: str, *, duration_ms=4000,
                     role="audio.primary_input", sample_rate=16000,
                     channels=1, volume_percent=None) -> dict[str, Any]:
        arguments = {
            "media_owner": media_owner, "role": role, "duration_ms": duration_ms,
            "sample_rate": sample_rate, "channels": channels,
        }
        if volume_percent is not None:
            arguments["volume_percent"] = volume_percent
        return self._call("audio.capture_clip", arguments, timeout_ms=20_000)

    def capture_turn(self, media_owner: str, *, role="audio.primary_input",
                     sample_rate=16000, channels=1, volume_percent=None,
                     max_wait_ms=8000, max_turn_ms=15000,
                     end_silence_ms=700, pre_roll_ms=300) -> dict[str, Any]:
        arguments = {
            "media_owner": media_owner, "role": role, "sample_rate": sample_rate,
            "channels": channels, "max_wait_ms": max_wait_ms,
            "max_turn_ms": max_turn_ms, "end_silence_ms": end_silence_ms,
            "pre_roll_ms": pre_roll_ms,
        }
        if volume_percent is not None:
            arguments["volume_percent"] = volume_percent
        return self._call("audio.capture_turn", arguments, timeout_ms=30_000)

    def start_playback(self, media_id: str, media_owner: str, *,
                       role="audio.primary_output", volume_percent=None) -> dict[str, Any]:
        arguments = {"media_id": media_id, "media_owner": media_owner, "role": role}
        if volume_percent is not None:
            arguments["volume_percent"] = volume_percent
        return self._call("audio.start_playback", arguments, timeout_ms=5_000)

    def playback_status(self, playback_id: str) -> dict[str, Any]:
        return self._call("audio.playback_status", {"playback_id": playback_id})

    def start_stream_playback(self, media_id: str, media_owner: str, *, stream_key: str,
                              role="audio.primary_output", volume_percent=None,
                              resolve_only=False) -> dict[str, Any]:
        arguments = {"media_id": media_id, "media_owner": media_owner,
                     "stream_key": stream_key, "role": role}
        if volume_percent is not None:
            arguments["volume_percent"] = volume_percent
        if resolve_only:
            arguments["resolve_only"] = resolve_only
        return self._call("audio.start_stream_playback", arguments, timeout_ms=5_000)

    def append_stream_playback(self, playback_id: str, media_id: str,
                               media_owner: str, segment_index: int) -> dict[str, Any]:
        return self._call("audio.append_stream_playback", {
            "playback_id": playback_id, "media_id": media_id,
            "media_owner": media_owner, "segment_index": segment_index,
        }, timeout_ms=5_000)

    def finish_stream_playback(self, playback_id: str, media_owner: str) -> dict[str, Any]:
        return self._call("audio.finish_stream_playback", {
            "playback_id": playback_id, "media_owner": media_owner,
        }, timeout_ms=2_000)

    def stop_playback(self, playback_id: str) -> dict[str, Any]:
        return self._call("audio.stop_playback", {"playback_id": playback_id}, timeout_ms=5_000)

    def interaction_status(self) -> dict[str, Any]:
        return self._call("audio.interaction_status", {})
