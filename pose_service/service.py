#!/usr/bin/env python3
"""Local-only MediaPipe Pose runner with an annotated ephemeral preview."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import statistics
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pose_service.metrics import landmark_quality, landmark_valid, posture_features


LOG = logging.getLogger("vq.pose")
BOUNDARY = "vq-pose"
MODEL_NAME = "mediapipe-pose-landmarker-lite-f16@2026-09-04"
MODEL_SHA256 = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"
EDGES = (
    (0, 7), (0, 8), (7, 11), (8, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
)


class PoseRuntime:
    """Own one model instance and retain only aggregate metrics/latest JPEG."""

    def __init__(self, model_path, camera_url, *, rotation=180, sample_fps=10.0):
        self.model_path = model_path
        self.camera_url = camera_url
        self.rotation = rotation
        self.sample_fps = sample_fps
        self.condition = threading.Condition()
        self.latest = None
        self.sequence = 0
        self.last_frame_at = None
        self.started_at = time.monotonic()
        self.error = None
        self.latest_features = None
        self.closed = False
        self.samples = deque(maxlen=300)
        self.thread = threading.Thread(target=self._run, name="vq-pose-inference", daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join(timeout=5)

    def status(self):
        with self.condition:
            samples = list(self.samples)
            age = None if self.last_frame_at is None else time.monotonic() - self.last_frame_at
            error = self.error
            sequence = self.sequence
            features = self.latest_features
        total = len(samples)
        latencies = [item[3] for item in samples]
        detected = sum(item[0] for item in samples)
        upper = sum(item[1] for item in samples)
        full = sum(item[2] for item in samples)
        return {
            "ready": bool(total and age is not None and age < 3.0),
            "model": MODEL_NAME,
            "model_sha256": MODEL_SHA256,
            "runtime": f"mediapipe-{mp.__version__}",
            "backend": "cpu-xnnpack",
            "npu_accelerated": False,
            "sequence": sequence,
            "window_frames": total,
            "pose_success_rate": round(detected / total, 3) if total else 0.0,
            "upper_body_success_rate": round(upper / total, 3) if total else 0.0,
            "full_core_success_rate": round(full / total, 3) if total else 0.0,
            "latency_ms": {
                "median": round(statistics.median(latencies), 1) if latencies else None,
                "mean": round(statistics.mean(latencies), 1) if latencies else None,
                "p95": round(sorted(latencies)[math.ceil(total * .95) - 1], 1) if latencies else None,
            },
            "last_frame_age_ms": round(age * 1000) if age is not None else None,
            "sample_fps": self.sample_fps,
            "rotation": self.rotation,
            "privacy": {"raw_frames_retained": False, "landmarks_exposed": False},
            "posture_features": features,
            "error": error,
        }

    def wait_frame(self, after=0, timeout=3.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.closed and (self.latest is None or self.sequence <= after):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            return None if self.latest is None else (self.sequence, self.latest)

    def frame(self):
        """Return one immutable latest-frame copy for non-streaming clients."""
        with self.condition:
            return self.latest

    def _rotate(self, frame):
        if self.rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _publish(self, frame, detected, quality, latency_ms, features):
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 86])
        if not ok:
            return
        with self.condition:
            self.latest = encoded.tobytes()
            self.sequence += 1
            self.last_frame_at = time.monotonic()
            self.error = None
            self.latest_features = features
            self.samples.append((int(detected), int(quality["upper_body_valid"]),
                                 int(quality["full_core_valid"]), latency_ms))
            self.condition.notify_all()

    def _annotate(self, frame, landmarks):
        height, width = frame.shape[:2]
        for a, b in EDGES:
            if landmark_valid(landmarks[a]) and landmark_valid(landmarks[b]):
                pa, pb = landmarks[a], landmarks[b]
                cv2.line(frame, (int(pa.x * width), int(pa.y * height)),
                         (int(pb.x * width), int(pb.y * height)), (189, 225, 104), 3)
        for point in landmarks:
            if landmark_valid(point):
                cv2.circle(frame, (int(point.x * width), int(point.y * height)),
                           4, (107, 201, 255), -1)
        return frame

    def _run(self):
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.35,
            min_pose_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        started = time.monotonic()
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while not self.closed:
                capture = cv2.VideoCapture(self.camera_url)
                if not capture.isOpened():
                    self._set_error("camera_stream_unavailable")
                    time.sleep(1)
                    continue
                try:
                    next_inference_at = 0.0
                    while not self.closed:
                        ok, frame = capture.read()
                        if not ok:
                            self._set_error("camera_stream_disconnected")
                            break
                        captured_at = time.monotonic()
                        # Always consume the MJPEG stream. If a source is faster
                        # than the inference budget, discard surplus frames here
                        # instead of letting TCP/OpenCV build a stale-frame queue.
                        if captured_at < next_inference_at:
                            continue
                        next_inference_at = captured_at + 1.0 / self.sample_fps
                        frame = self._rotate(frame)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        before = time.perf_counter()
                        result = landmarker.detect_for_video(
                            image, int((time.monotonic() - started) * 1000),
                        )
                        latency = (time.perf_counter() - before) * 1000
                        detected = bool(result.pose_landmarks)
                        quality = landmark_quality(result.pose_landmarks[0] if detected else None)
                        world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
                        features = posture_features(
                            result.pose_landmarks[0] if detected else None, world,
                        )
                        if detected:
                            frame = self._annotate(frame, result.pose_landmarks[0])
                        self._publish(frame, detected, quality, latency, features)
                except Exception as exc:
                    LOG.exception("pose inference loop failed")
                    self._set_error(type(exc).__name__)
                finally:
                    capture.release()
                if not self.closed:
                    time.sleep(0.5)

    def _set_error(self, value):
        with self.condition:
            self.error = value
            self.condition.notify_all()


class Handler(BaseHTTPRequestHandler):
    server_version = "VQPose/0.1"

    @property
    def runtime(self):
        return self.server.runtime

    def log_message(self, fmt, *args):
        LOG.debug(fmt, *args)

    def do_GET(self):
        if self.path in {"/healthz", "/status"}:
            value = self.runtime.status()
            status = HTTPStatus.OK if self.path == "/status" or value["ready"] else HTTPStatus.SERVICE_UNAVAILABLE
            self._json(status, value)
        elif self.path == "/live.mjpg":
            self._stream()
        elif self.path == "/frame.jpg":
            frame = self.runtime.frame()
            if frame is None:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "frame unavailable"})
            else:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(frame)
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _stream(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        sequence = 0
        try:
            while True:
                value = self.runtime.wait_frame(sequence)
                if value is None:
                    continue
                sequence, frame = value
                self.wfile.write(
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame)}\r\n\r\n".encode()
                )
                self.wfile.write(frame + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _json(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--model", required=True)
    parser.add_argument("--camera-url", default="http://127.0.0.1:8765/api/camera/live.mjpg")
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument("--sample-fps", type=float, default=10.0,
                        help="maximum inference FPS; camera bytes are always drained")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1"}:
        parser.error("pose service may bind only to loopback")
    if not 0.5 <= args.sample_fps <= 10:
        parser.error("sample-fps must be between 0.5 and 10")
    try:
        with open(args.model, "rb") as model_file:
            digest = hashlib.file_digest(model_file, "sha256").hexdigest()
    except OSError as exc:
        parser.error(f"cannot read pose model: {exc}")
    if digest != MODEL_SHA256:
        parser.error("pose model checksum mismatch")
    logging.basicConfig(level=logging.INFO)
    runtime = PoseRuntime(args.model, args.camera_url, rotation=args.rotation, sample_fps=args.sample_fps)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.runtime = runtime
    runtime.start()
    try:
        server.serve_forever()
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
