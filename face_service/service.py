"""Headless multi-source face detector and ephemeral tracker."""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from camera_stream_service import CameraStreamClient
from event_service import EventClient

from .tracker import FaceTracker


LOG = logging.getLogger("vq.face")
DEFAULT_MODEL = Path.home() / ".local/share/vq-ai/models/face/yunet/face_detection_yunet_2026may.onnx"
STATIC_ROOT = Path(__file__).with_name("static")


class YuNetDetector:
    def __init__(self, model_path, *, threshold=0.75, rotation=180):
        import cv2
        # Two camera workers share the host. One OpenCV thread per detector
        # avoids nested parallelism stealing cores from Pose, audio and Agent.
        cv2.setNumThreads(1)
        self.cv2 = cv2
        self.rotation = rotation
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), threshold, 0.3, 5000)

    def detect(self, jpeg):
        import numpy as np
        cv2 = self.cv2
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("invalid_jpeg")
        if self.rotation == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif self.rotation == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _unused, rows = self.detector.detect(image)
        result = []
        if rows is not None:
            for row in rows:
                x, y, box_width, box_height = [float(value) for value in row[:4]]
                normal_x = max(0.0, min(1.0, x / width))
                normal_y = max(0.0, min(1.0, y / height))
                result.append({
                    "bbox": (
                        normal_x, normal_y,
                        max(0.0, min(1.0 - normal_x, box_width / width)),
                        max(0.0, min(1.0 - normal_y, box_height / height)),
                    ),
                    "score": float(row[14]),
                })
        return result


class FaceSourceWorker:
    def __init__(self, source, detector, callback, *, fps=5.0,
                 client_factory=CameraStreamClient, clock=time.monotonic):
        self.source = source
        self.detector = detector
        self.callback = callback
        self.fps = fps
        self.clock = clock
        self.client = client_factory(consumer="face", source=source, rotation=0)
        self.tracker = FaceTracker(prefix=source.rsplit(".", 1)[-1])
        self.sequence = 0
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"vq-face-{source}", daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.closed.set()
        self.thread.join(timeout=5)

    def _run(self):
        interval = 1.0 / self.fps
        while not self.closed.is_set():
            started = self.clock()
            try:
                value = self.client.wait_frame(self.sequence, timeout=3)
                if value is None:
                    now = self.clock()
                    events = self.tracker.update([], now)
                    self.callback(self.source, None, self.tracker.snapshot(now),
                                  "frame_unavailable", events)
                else:
                    self.sequence, jpeg = value
                    detections = self.detector.detect(jpeg)
                    now = self.clock()
                    events = self.tracker.update(detections, now)
                    state = self.tracker.snapshot(now)
                    latency_ms = round((now - started) * 1000, 2)
                    self.callback(self.source, self.sequence, state, None, events,
                                  latency_ms=latency_ms)
            except Exception as exc:
                LOG.warning("face source failed source=%s error=%s", self.source, type(exc).__name__)
                now = self.clock()
                events = self.tracker.update([], now)
                self.callback(self.source, None, self.tracker.snapshot(now),
                              type(exc).__name__, events)
            self.closed.wait(max(0.0, interval - (self.clock() - started)))


class FaceService:
    def __init__(self, sources, detector_factory, *, fps=5.0, events=None,
                 worker_factory=FaceSourceWorker, wall_clock=time.time,
                 camera_client_factory=CameraStreamClient):
        self.lock = threading.Lock()
        self.wall_clock = wall_clock
        self.events = events
        self.camera_client_factory = camera_client_factory
        self.debug_frame_requests = 0
        self.last_debug_frame_at = None
        self.states = {source: {
            "source": source, "ready": False, "sequence": 0,
            "face_count": 0, "candidate_count": 0, "tracks": [],
            "latency_ms": None, "last_frame_at": None, "error": None,
        } for source in sources}
        self.workers = [worker_factory(source, detector_factory(source), self._record, fps=fps)
                        for source in sources]

    def start(self):
        for worker in self.workers:
            worker.start()

    def close(self):
        for worker in self.workers:
            worker.close()

    def _record(self, source, sequence, state, error, events, *, latency_ms=None):
        now = self.wall_clock()
        with self.lock:
            current = self.states[source]
            if state is not None:
                current.update(state)
                current.update(ready=error is None, latency_ms=latency_ms, error=error)
                if sequence is not None:
                    current.update(sequence=sequence, last_frame_at=now)
            else:
                current.update(ready=False, error=error)
        for event in events:
            self._publish(source, event)

    def _publish(self, source, event):
        if self.events is None:
            return
        topic = "perception.face.entered" if event["type"] == "entered" else "perception.face.left"
        payload = {"camera_role": source, **{key: value for key, value in event.items()
                                            if key != "type"}}
        try:
            self.events.publish(topic, "face-service", payload, privacy="local-private")
        except Exception as exc:
            LOG.warning("face event failed topic=%s error=%s", topic, type(exc).__name__)

    def status(self):
        with self.lock:
            sources = [dict(value, tracks=[dict(track) for track in value["tracks"]])
                       for value in self.states.values()]
            debug_age = None if self.last_debug_frame_at is None else max(
                0.0, self.wall_clock() - self.last_debug_frame_at)
            debug = {
                "active": debug_age is not None and debug_age < 1.0,
                "request_count": self.debug_frame_requests,
                "last_request_seconds_ago": round(debug_age, 2) if debug_age is not None else None,
            }
        return {
            "ready": any(value["ready"] for value in sources),
            "model": "opencv-yunet-2026may",
            "mode": "headless",
            "identity_recognition": False,
            "debug_preview": debug,
            "sources": sources,
            "privacy": {
                "raw_frames_retained": False,
                "biometric_templates_retained": False,
                "track_ids": "ephemeral-per-process",
            },
        }

    def debug_frame(self, source):
        with self.lock:
            if source not in self.states:
                raise KeyError(source)
        client = self.camera_client_factory(
            consumer="face-debug", source=source, rotation=0)
        value = client.wait_frame(0, timeout=3)
        if value is None:
            raise RuntimeError("frame_unavailable")
        with self.lock:
            self.debug_frame_requests += 1
            self.last_debug_frame_at = self.wall_clock()
        return value


class Handler(BaseHTTPRequestHandler):
    server_version = "VQFace/0.1"
    def log_message(self, *_args): pass
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/debug", "/debug/"}:
            return self._static("debug.html", "text/html; charset=utf-8")
        if path == "/debug/app.js":
            return self._static("app.js", "text/javascript; charset=utf-8")
        if path == "/debug/style.css":
            return self._static("style.css", "text/css; charset=utf-8")
        if path == "/debug/frame.jpg":
            from urllib.parse import parse_qs
            source = parse_qs(parsed.query).get("source", [""])[0]
            try:
                sequence, frame = self.server.face.debug_frame(source)
            except KeyError:
                return self._json(HTTPStatus.NOT_FOUND, {"error": "source_not_found"})
            except Exception:
                return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "frame_unavailable"})
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("X-VQ-Sequence", str(sequence))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return self.wfile.write(frame)
        if path not in {"/status", "/healthz"}:
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        value = self.server.face.status()
        code = HTTPStatus.OK if path == "/status" or value["ready"] else HTTPStatus.SERVICE_UNAVAILABLE
        self._json(code, value)
    def _json(self, status, value):
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
    def _static(self, name, content_type):
        try:
            body = (STATIC_ROOT / name).read_bytes()
        except OSError:
            return self._json(HTTPStatus.NOT_FOUND, {"error": "asset_not_found"})
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self'; script-src 'self'; "
                         "style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8794)
    parser.add_argument("--source", action="append")
    parser.add_argument("--model", default=os.environ.get("VQ_FACE_MODEL", str(DEFAULT_MODEL)))
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--rotation", type=int, default=180, choices=(0, 90, 180, 270))
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1"}:
        parser.error("face service is loopback only")
    if not 1 <= args.fps <= 15:
        parser.error("fps must be between 1 and 15")
    model_path = Path(args.model).expanduser()
    if not model_path.is_file():
        parser.error(f"face model not found: {model_path}")
    sources = args.source or ["camera.front", "camera.environment"]
    logging.basicConfig(level=logging.INFO)
    detector_factory = lambda _source: YuNetDetector(
        model_path, threshold=0.75, rotation=args.rotation)
    face = FaceService(sources, detector_factory, fps=args.fps, events=EventClient(timeout=1))
    face.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.face = face
    try:
        server.serve_forever()
    finally:
        face.close()


if __name__ == "__main__":
    main()
