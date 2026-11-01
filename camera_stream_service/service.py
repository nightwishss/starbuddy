#!/usr/bin/env python3
"""Multi-camera UVC broker with one owner and latest-frame fan-out per source."""

from __future__ import annotations

import argparse
import json
import logging
import re
import select
import subprocess
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from device_management import HardwareProfileManager


LOG = logging.getLogger("vq.camera-stream")
BOUNDARY = "vq-camera-stream"
MAX_FRAME_BYTES = 2 * 1024 * 1024
CONSUMER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


class CameraUnavailableError(RuntimeError):
    pass


class CameraRecoveryPolicy:
    """Per-source bounded exponential retry state without cross-camera coupling."""

    def __init__(self, initial_seconds=1.0, maximum_seconds=30.0):
        if initial_seconds <= 0 or maximum_seconds < initial_seconds:
            raise ValueError("invalid camera recovery backoff")
        self.initial_seconds = float(initial_seconds)
        self.maximum_seconds = float(maximum_seconds)
        self.next_delay_seconds = self.initial_seconds
        self.consecutive_failures = 0

    def failure(self) -> float:
        delay = self.next_delay_seconds
        self.consecutive_failures += 1
        self.next_delay_seconds = min(self.maximum_seconds, delay * 2)
        return delay

    def frame_received(self) -> None:
        self.consecutive_failures = 0
        self.next_delay_seconds = self.initial_seconds


class CameraStream:
    def __init__(self, profile_manager=None, *, role="camera.environment",
                 width=640, height=480, source_fps=30, popen=subprocess.Popen,
                 stale_seconds=3.0, retry_initial_seconds=1.0,
                 retry_max_seconds=30.0, clock=time.monotonic):
        self.profile_manager = profile_manager or HardwareProfileManager()
        self.role = role
        self.width = width
        self.height = height
        self.source_fps = source_fps
        self.popen = popen
        self.clock = clock
        self.stale_seconds = stale_seconds
        self.recovery = CameraRecoveryPolicy(retry_initial_seconds, retry_max_seconds)
        self.condition = threading.Condition()
        self.latest = None
        self.sequence = 0
        self.last_frame_at = None
        self.error = None
        self.source_id = None
        self.stable_id = None
        self.endpoint = None
        self.last_source_id = None
        self.last_stable_id = None
        self.last_endpoint = None
        self.closed = False
        self.process = None
        self.capture_active = False
        self.subscribers = {}
        self.frame_lease_until = 0.0
        self.source_frames = 0
        self.sent_frames = 0
        self.dropped_frames = 0
        self.restarts = 0
        self.next_retry_at = None
        thread_role = role.replace(".", "-")
        self.thread = threading.Thread(target=self._run, name=f"vq-camera-{thread_role}", daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        with self.condition:
            self.closed = True
            process = self.process
            self.condition.notify_all()
        self._stop_process(process)
        self.thread.join(timeout=4)

    def register(self, consumer, fps):
        token = uuid.uuid4().hex
        with self.condition:
            self.subscribers[token] = {"consumer": consumer, "fps": fps, "sent": 0, "dropped": 0}
            self.condition.notify_all()
        return token

    def unregister(self, token):
        with self.condition:
            value = self.subscribers.pop(token, None)
            if value:
                self.sent_frames += value["sent"]
                self.dropped_frames += value["dropped"]
            self.condition.notify_all()

    def request_frame(self, seconds=2.0):
        with self.condition:
            self.frame_lease_until = max(self.frame_lease_until, self.clock() + seconds)
            self.condition.notify_all()

    def update_resolution(self, resolved):
        device = resolved.get("device") if resolved.get("status") == "resolved" else None
        with self.condition:
            self.source_id = device.get("device_id") if device else None
            self.stable_id = device.get("stable_id") if device else None
            self.endpoint = device.get("endpoint") if device else None
            if device:
                self.last_source_id = self.source_id
                self.last_stable_id = self.stable_id
                self.last_endpoint = self.endpoint

    def wait_frame(self, after=0, timeout=3.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.closed and (self.latest is None or self.sequence <= after):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            return None if self.latest is None else (self.sequence, self.latest, self.last_frame_at)

    def account(self, token, previous, current):
        with self.condition:
            value = self.subscribers.get(token)
            if value is None:
                return
            value["sent"] += 1
            value["dropped"] += max(0, current - previous - 1)

    def status(self):
        with self.condition:
            age = None if self.last_frame_at is None else self.clock() - self.last_frame_at
            active = list(self.subscribers.values())
            return {
                "ready": bool(self.latest and age is not None and age < self.stale_seconds),
                "connected": self.source_id is not None,
                "capture_active": self.capture_active,
                "role": self.role, "sequence": self.sequence,
                "source_id": self.source_id, "stable_id": self.stable_id,
                "endpoint": self.endpoint,
                "last_source_id": self.last_source_id,
                "last_stable_id": self.last_stable_id,
                "last_endpoint": self.last_endpoint,
                "last_frame_age_ms": round(age * 1000) if age is not None else None,
                "width": self.width, "height": self.height,
                "source_fps": self.source_fps, "source_frames": self.source_frames,
                "subscriber_count": len(active),
                "subscribers": [{"consumer": x["consumer"], "fps": x["fps"],
                                 "sent": x["sent"], "dropped": x["dropped"]} for x in active],
                "sent_frames": self.sent_frames + sum(x["sent"] for x in active),
                "dropped_frames": self.dropped_frames + sum(x["dropped"] for x in active),
                "restarts": self.restarts, "error": self.error,
                "consecutive_failures": self.recovery.consecutive_failures,
                "next_retry_ms": None if self.next_retry_at is None else round(
                    max(0.0, self.next_retry_at - self.clock()) * 1000),
                "ownership": "single-owner", "queue_policy": "latest-frame-drop-old",
            }

    def _needed_locked(self):
        return bool(self.subscribers) or self.clock() < self.frame_lease_until

    def _run(self):
        while True:
            with self.condition:
                while not self.closed and not self._needed_locked():
                    self.latest = None
                    self.condition.wait()
                if self.closed:
                    return
            process = None
            try:
                resolved = self.profile_manager.resolve(self.role)
                device_info = resolved.get("device") if resolved.get("status") == "resolved" else None
                if not isinstance(device_info, dict) or not device_info.get("endpoint"):
                    raise CameraUnavailableError("camera role is not currently resolved")
                device = device_info["endpoint"]
                with self.condition:
                    self.source_id = device_info.get("device_id")
                    self.stable_id = device_info.get("stable_id")
                    self.endpoint = device
                    self.last_source_id = self.source_id
                    self.last_stable_id = self.stable_id
                    self.last_endpoint = self.endpoint
                settings = resolved.get("settings") or {}
                source_fps = settings.get("source_fps", self.source_fps)
                if isinstance(source_fps, int) and 1 <= source_fps <= 60:
                    self.source_fps = source_fps
                command = [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-f", "v4l2", "-input_format", "mjpeg",
                    "-video_size", f"{self.width}x{self.height}",
                    "-framerate", str(self.source_fps), "-i", device,
                    "-an", "-c:v", "copy", "-f", "image2pipe", "pipe:1",
                ]
                process = self.popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
                with self.condition:
                    self.process = process
                    self.capture_active = True
                try:
                    self._read_frames(process.stdout)
                finally:
                    # Release the device and pipe BEFORE retry backoff. Otherwise
                    # a stalled ffmpeg keeps consuming resources for up to 30 s.
                    self._stop_process(process)
                    process = None
                    with self.condition:
                        self.process = None
                        self.capture_active = False
                with self.condition:
                    still_needed = not self.closed and self._needed_locked()
                if still_needed:
                    raise EOFError("camera capture ended without a complete frame")
            except Exception as exc:
                with self.condition:
                    if self.closed:
                        return
                    if isinstance(exc, TimeoutError):
                        self.error = "frame_stale"
                    elif isinstance(exc, EOFError):
                        self.error = "capture_ended"
                    else:
                        self.error = type(exc).__name__
                    self.latest = None
                    self.capture_active = False
                    self.source_id = None
                    self.stable_id = None
                    self.endpoint = None
                    self.restarts += 1
                    delay = self.recovery.failure()
                    self.next_retry_at = self.clock() + delay
                LOG.warning(
                    "camera_source_restart role=%s source_id=%s error=%s "
                    "retry_seconds=%.1f failures=%d",
                    self.role, self.last_source_id, self.error, delay,
                    self.recovery.consecutive_failures,
                )
                with self.condition:
                    while not self.closed and self.clock() < self.next_retry_at:
                        self.condition.wait(self.next_retry_at - self.clock())
            finally:
                self._stop_process(process)
                with self.condition:
                    self.process = None
                    self.capture_active = False
                    self.next_retry_at = None

    @staticmethod
    def _stop_process(process):
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
                except ProcessLookupError:
                    process.wait(timeout=2)
        finally:
            for name in ("stdout", "stderr"):
                pipe = getattr(process, name, None)
                if pipe is not None:
                    pipe.close()

    def _read_frames(self, stream):
        buffer = bytearray()
        frame_deadline = self.clock() + self.stale_seconds
        recovered = False
        while True:
            with self.condition:
                if self.closed or not self._needed_locked():
                    return
            # A disconnected UVC device can leave ffmpeg alive without writing.
            # On real pipes, bound the wait so the owner can restart and rediscover.
            remaining = frame_deadline - self.clock()
            if remaining <= 0:
                raise TimeoutError("camera complete frame stalled")
            try:
                readable, _, _ = select.select([stream], [], [], remaining)
            except (TypeError, ValueError, OSError):
                readable = [stream]
            if not readable:
                with self.condition:
                    self.error = "frame_stale"
                raise TimeoutError("camera frame source stalled")
            chunk = stream.read(64 * 1024)
            if not chunk:
                return
            buffer.extend(chunk)
            while True:
                start = buffer.find(b"\xff\xd8")
                if start < 0:
                    if len(buffer) > 1:
                        del buffer[:-1]
                    break
                end = buffer.find(b"\xff\xd9", start + 2)
                if end < 0:
                    if start:
                        del buffer[:start]
                    if len(buffer) > MAX_FRAME_BYTES:
                        buffer.clear()
                    break
                frame = bytes(buffer[start:end + 2])
                del buffer[:end + 2]
                if len(frame) > MAX_FRAME_BYTES:
                    continue
                with self.condition:
                    self.latest = frame
                    self.sequence += 1
                    self.source_frames += 1
                    self.last_frame_at = self.clock()
                    frame_deadline = self.last_frame_at + self.stale_seconds
                    if not recovered and self.recovery.consecutive_failures:
                        LOG.info("camera_source_recovered role=%s failures=%d sequence=%d",
                                 self.role, self.recovery.consecutive_failures, self.sequence)
                    recovered = True
                    self.error = None
                    self.recovery.frame_received()
                    self.next_retry_at = None
                    self.condition.notify_all()


class CameraBroker:
    """Own all configured camera roles and route consumers to one source."""

    def __init__(self, profile_manager=None, *, stream_factory=CameraStream):
        self.profile_manager = profile_manager or HardwareProfileManager()
        profile = self.profile_manager.profile()
        roles = sorted(role for role, spec in profile["roles"].items()
                       if spec.get("kind") == "camera")
        self.default_role = "camera.environment" if "camera.environment" in roles else roles[0]
        self.streams = {
            role: stream_factory(self.profile_manager, role=role) for role in roles
        }

    def start(self):
        for stream in self.streams.values():
            stream.start()

    def close(self):
        for stream in self.streams.values():
            stream.close()

    def select(self, source=None):
        key = source or self.default_role
        if key in self.streams:
            return self.streams[key]
        self._refresh_resolutions()
        for stream in self.streams.values():
            status = stream.status()
            if key in {status.get("source_id"), status.get("stable_id")}:
                return stream
        return None

    def _refresh_resolutions(self):
        try:
            roles = self.profile_manager.resolve_all()["roles"]
        except (KeyError, TypeError, AttributeError):
            return
        for role, stream in self.streams.items():
            if role in roles:
                stream.update_resolution(roles[role])

    def status(self):
        self._refresh_resolutions()
        sources = [stream.status() for stream in self.streams.values()]
        return {
            "ready": any(item["ready"] for item in sources),
            "source_count": len(sources),
            "connected_source_count": sum(bool(item["connected"]) for item in sources),
            "ready_source_count": sum(bool(item["ready"]) for item in sources),
            "default_role": self.default_role,
            "sources": sources,
            "ownership": "single-owner-per-physical-source",
            "queue_policy": "latest-frame-drop-old",
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "VQCameraStream/0.2"

    def log_message(self, fmt, *args):
        LOG.debug(fmt, *args)

    @property
    def broker(self):
        return self.server.broker

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        requested_source = query.get("source", [None])[0]
        if parsed.path == "/sources":
            return self._json(HTTPStatus.OK, self.broker.status())
        if parsed.path in {"/healthz", "/status"}:
            stream = self.broker.select(requested_source)
            if stream is None:
                return self._json(HTTPStatus.NOT_FOUND, {"error": "source_not_found"})
            value = stream.status()
            value["available_sources"] = len(self.broker.streams)
            code = HTTPStatus.OK if parsed.path == "/status" or value["ready"] else HTTPStatus.SERVICE_UNAVAILABLE
            return self._json(code, value)
        stream = self.broker.select(requested_source)
        if stream is None:
            return self._json(HTTPStatus.NOT_FOUND, {"error": "source_not_found"})
        consumer = query.get("consumer", ["anonymous"])[0]
        if not CONSUMER_RE.fullmatch(consumer):
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_consumer"})
        if parsed.path == "/frame.jpg":
            try:
                after = int(query.get("after", ["0"])[0])
            except ValueError:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_sequence"})
            if after < 0:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_sequence"})
            stream.request_frame()
            value = stream.wait_frame(after=after, timeout=3)
            if value is None:
                return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "frame_unavailable"})
            sequence, frame, _created = value
            return self._frame(sequence, frame)
        if parsed.path == "/live.mjpg":
            try:
                fps = float(query.get("fps", ["6"])[0])
            except ValueError:
                fps = 0
            if not 1 <= fps <= 30:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_fps"})
            return self._live(stream, consumer, fps)
        return self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def _live(self, stream, consumer, fps):
        token = stream.register(consumer, fps)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        cursor = 0
        last_sent = 0
        next_send = 0.0
        try:
            while True:
                value = stream.wait_frame(cursor, timeout=4)
                if value is None:
                    continue
                current, frame, _created = value
                cursor = current
                now = time.monotonic()
                if now < next_send:
                    continue
                next_send = now + 1.0 / fps
                stream.account(token, last_sent, current)
                last_sent = current
                self.wfile.write(
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame)}\r\n"
                    f"X-VQ-Sequence: {current}\r\n\r\n".encode("ascii")
                )
                self.wfile.write(frame + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        finally:
            stream.unregister(token)

    def _frame(self, sequence, frame):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.send_header("X-VQ-Sequence", str(sequence))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(frame)

    def _json(self, status, value):
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1"}:
        parser.error("camera stream service may bind only to loopback")
    logging.basicConfig(level=logging.INFO)
    broker = CameraBroker()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.broker = broker
    broker.start()
    try:
        server.serve_forever()
    finally:
        broker.close()


if __name__ == "__main__":
    main()
