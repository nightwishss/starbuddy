"""Owner-only Unix-socket API for unified speech/audio output."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from event_service import EventClient
from media_service import MediaClient
from runtime.client import RuntimeClient
from sdk.audio import AudioClient
from sdk.speech import SpeechClient

from .arbiter import OutputArbiter, OutputError
from .client import DEFAULT_SOCKET


LOG = logging.getLogger("vq.output")


class RuntimeOutputEngine:
    def __init__(self):
        runtime = RuntimeClient(timeout=35.0)
        self.speech = SpeechClient(runtime)
        self.audio = AudioClient(RuntimeClient(timeout=8.0))
        self.speech_status = SpeechClient(RuntimeClient(timeout=2.0))
        self.media = MediaClient(timeout=5.0)

    def validate_guard(self, guard, job_id):
        from perception_context_service.client import PerceptionContextClient
        try:
            return bool(PerceptionContextClient(timeout=1.2).call(
                "fusion.validate", guard=guard, job_id=job_id).get("allowed"))
        except Exception:
            return False

    def reserve_output(self, owner):
        from control_plane.client import ControlPlaneClient
        return ControlPlaneClient(timeout=.2).acquire(
            owner, {"audio_output": 1}, "posture-reminder", priority=20, ttl_seconds=60)["lease_id"]

    def release_output(self, lease_id, owner):
        from control_plane.client import ControlPlaneClient
        return ControlPlaneClient(timeout=.2).release(lease_id, owner)

    def synthesize(self, text, owner, *, speed=1.0):
        return self.speech.synthesize(text, owner, **({"speed": speed} if speed != 1 else {}))

    def start_playback(self, media_id, owner, volume_percent=70):
        return self.audio.start_playback(media_id, owner, role="audio.primary_output",
                                         volume_percent=volume_percent)

    def streaming_available(self):
        try:
            status = self.speech_status.status()
            backends = status.get("tts_backends", {})
            backend = backends.get("default")
            # Both adapters produce owned PCM WAVs for the same Runtime player.
            # Do not mix a Chinese default with a different English voice. STT
            # readiness / worker residency is not required for text-only speech;
            # native startup and admission remain the synthesizer's responsibility.
            return (status.get("tts") is True and status.get("tts_bilingual") is True
                    and backend in {"melo-zh-en", "melo-qnn-htp"}
                    and backends.get("mixed_zh_en") == backend)
        except Exception:
            return False

    def start_stream_playback(self, media_id, owner, *, stream_key, volume_percent=70, resolve_only=False):
        return self.audio.start_stream_playback(media_id, owner, stream_key=stream_key,
                                                volume_percent=volume_percent, resolve_only=resolve_only)

    def append_stream_playback(self, playback_id, media_id, owner, segment_index):
        return self.audio.append_stream_playback(playback_id, media_id, owner, segment_index)

    def finish_stream_playback(self, playback_id, owner):
        return self.audio.finish_stream_playback(playback_id, owner)

    def playback_status(self, playback_id):
        return self.audio.playback_status(playback_id)

    def stop_playback(self, playback_id):
        return self.audio.stop_playback(playback_id)

    def release(self, media_id, owner):
        return self.media.call("release", owner=owner, media_id=media_id)


class OutputService:
    def __init__(self, arbiter):
        self.arbiter = arbiter

    async def dispatch(self, request):
        if not isinstance(request, dict):
            raise OutputError("invalid_request", "request must be an object")
        op = request.get("operation")
        values = {k: v for k, v in request.items() if k != "operation"}
        if op == "say":
            return self.arbiter.say(**values)
        if op == "play_media":
            return self.arbiter.play_media(**values)
        if op == "status":
            return self.arbiter.status(**values)
        if op == "cancel":
            return self.arbiter.cancel(**values)
        if op == "system.status" and not values:
            return self.arbiter.system_status()
        raise OutputError("unknown_operation", "unknown operation")

    async def handle(self, reader, writer):
        try:
            raw = await reader.readline()
            if not raw or len(raw) > 16 * 1024 or not raw.endswith(b"\n"):
                raise OutputError("invalid_request", "invalid request framing")
            response = {"ok": True, "result": await self.dispatch(json.loads(raw))}
        except OutputError as exc:
            response = {"ok": False, "code": exc.code, "error": str(exc)}
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as exc:
            response = {"ok": False, "code": "invalid_arguments", "error": str(exc)}
        except Exception as exc:
            LOG.exception("output request failed")
            response = {"ok": False, "code": "internal_error", "error": type(exc).__name__}
        writer.write((json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()


async def run(socket_path=DEFAULT_SOCKET):
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    try:
        events = EventClient(timeout=1.0)
        events.stats()
    except Exception:
        events = None
    arbiter = OutputArbiter(RuntimeOutputEngine(), events=events,
                           segmented_tts=os.environ.get("VQ_OUTPUT_SEGMENTED_TTS", "0") == "1")
    arbiter.start()
    server = await asyncio.start_unix_server(OutputService(arbiter).handle, path=socket_path)
    os.chmod(socket_path, 0o600)
    try:
        async with server:
            await server.serve_forever()
    finally:
        if not arbiter.close():
            LOG.error("output_shutdown_unconfirmed operator_runtime_check_required")
        path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(args.socket))


if __name__ == "__main__":
    main()
