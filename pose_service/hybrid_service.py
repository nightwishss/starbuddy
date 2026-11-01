#!/usr/bin/env python3
"""HTTP/annotation shell around the stateful detector + QNN pose worker."""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import threading
import time
import urllib.request
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import cv2
from observability.faults import FaultLog
from pose_service.metrics import landmark_quality, landmark_valid, posture_features
from pose_service.admission import PoseAdmission


LOG = logging.getLogger("vq.pose.hybrid")
BOUNDARY = "vq-pose-hybrid"
MODEL_NAME = "blazepose-cpu-detector-qnn-landmark-fp16@2026-09-08"
EDGES = ((0, 7), (0, 8), (7, 11), (8, 12), (11, 12), (11, 13), (13, 15),
         (12, 14), (14, 16), (11, 23), (12, 24), (23, 24))


class PoseRuntime:
    def __init__(self, camera_url, worker_url, *, rotation=180, sample_fps=10.0,
                 preview_ttl=2.0, admission=None):
        self.camera_url = camera_url
        self.worker_url = worker_url
        self.rotation = rotation
        self.sample_fps = sample_fps
        self.admission = admission if admission is not None else PoseAdmission()
        self.condition = threading.Condition()
        self.latest = None
        self.sequence = 0
        self.preview_sequence = 0
        self.preview_until = 0.0
        self.preview_ttl = preview_ttl
        self.last_preview_at = None
        self.last_frame_at = None
        self.error = None
        self.latest_features = None
        self.closed = False
        self.samples = deque(maxlen=300)
        self.detector_refreshes = 0
        self.npu_failures = 0
        self.pose_misses = 0
        self.faults = FaultLog(LOG)
        self.thread = threading.Thread(target=self._run, name="vq-pose-hybrid", daemon=True)

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
            preview_active = time.monotonic() < self.preview_until
            state = (self.error, self.sequence, self.latest_features,
                     self.detector_refreshes, self.npu_failures,
                     preview_active, self.preview_sequence, self.pose_misses)
        latencies = [item[3] for item in samples]
        total = len(samples)
        return {
            "ready": bool(total and age is not None and age < 3.0),
            "model": MODEL_NAME,
            "runtime": "ai-edge-litert+qnn-worker",
            "backend": "cpu-blazepose-detector+qnn-htp-landmark",
            "npu_accelerated": True,
            "sequence": state[1],
            "window_frames": total,
            "pose_success_rate": round(sum(x[0] for x in samples) / total, 3) if total else 0.0,
            "upper_body_success_rate": round(sum(x[1] for x in samples) / total, 3) if total else 0.0,
            "full_core_success_rate": round(sum(x[2] for x in samples) / total, 3) if total else 0.0,
            "latency_ms": {
                "median": round(statistics.median(latencies), 1) if latencies else None,
                "mean": round(statistics.mean(latencies), 1) if latencies else None,
                "p95": round(sorted(latencies)[math.ceil(total * .95) - 1], 1) if latencies else None,
            },
            "cpu_detector_refreshes": state[3],
            "npu_failures": state[4],
            "pose_misses": state[7],
            "last_frame_age_ms": round(age * 1000) if age is not None else None,
            "sample_fps": self.sample_fps,
            "rotation": self.rotation,
            "preview_active": state[5],
            "preview_sequence": state[6],
            "privacy": {"raw_frames_retained": False, "landmarks_exposed": False},
            "posture_features": state[2],
            "error": state[0],
            "resource_admission": self.admission.status(),
        }

    def frame(self):
        with self.condition:
            self._request_preview_locked()
            return self.latest

    def wait_frame(self, after=0, timeout=3.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            self._request_preview_locked()
            while not self.closed and (self.latest is None or self.preview_sequence <= after):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
                self._request_preview_locked()
            return None if self.latest is None else (self.preview_sequence, self.latest)

    def _request_preview_locked(self):
        self.preview_until = time.monotonic() + self.preview_ttl

    def _rotate(self, frame):
        if self.rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _infer_npu(self, frame):
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("frame_encode_failed")
        request = urllib.request.Request(
            self.worker_url, data=encoded.tobytes(), method="POST",
            headers={"Content-Type": "image/jpeg"},
        )
        with urllib.request.urlopen(request, timeout=2.0) as response:
            return json.load(response)

    def _try_infer_npu(self, frame):
        if not self.admission.admit():
            # Leave sequence/capture timestamp untouched. Old pose becomes
            # stale naturally, never a synthetic healthy/no-person sample.
            return None
        try:
            return self._infer_npu(frame)
        except Exception as exc:
            with self.condition:
                self.npu_failures += 1
                self.error = type(exc).__name__
                self.condition.notify_all()
            self.faults.record("npu_inference_failed", action="retry", error=exc)
            return None

    def _publish(self, frame, landmarks, detected, quality, latency, features):
        with self.condition:
            preview_active = time.monotonic() < self.preview_until
        encoded_bytes = None
        if preview_active:
            if detected:
                self._annotate(frame, landmarks)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 86])
            if ok:
                encoded_bytes = encoded.tobytes()
        with self.condition:
            self.sequence += 1
            self.last_frame_at = time.monotonic()
            self.error = None
            self.latest_features = features
            self.samples.append((int(detected), int(quality["upper_body_valid"]),
                                 int(quality["full_core_valid"]), latency))
            if encoded_bytes is not None:
                self.latest = encoded_bytes
                self.preview_sequence += 1
                self.last_preview_at = self.last_frame_at
            elif not preview_active:
                self.latest = None
            self.condition.notify_all()

    @staticmethod
    def _annotate(frame, landmarks):
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
                    now = time.monotonic()
                    if now < next_inference_at:
                        continue
                    next_inference_at = now + 1.0 / self.sample_fps
                    frame = self._rotate(frame)
                    before = time.perf_counter()
                    response = self._try_infer_npu(frame)
                    if response is None:
                        if not self.admission.enabled or self.admission.allowed:
                            time.sleep(min(0.25, 1.0 / self.sample_fps))
                        continue
                    latency = (time.perf_counter() - before) * 1000.0
                    if response.get("detector_ran"):
                        self.detector_refreshes += 1
                    detected = bool(response.get("detected") and response.get("landmarks"))
                    landmarks = [SimpleNamespace(**point) for point in response["landmarks"]] \
                        if detected else None
                    world = [SimpleNamespace(**point) for point in response["world_landmarks"]] \
                        if detected else None
                    quality = landmark_quality(landmarks)
                    features = posture_features(landmarks, world)
                    if not detected:
                        self.pose_misses += 1
                    self._publish(frame, landmarks, detected, quality, latency, features)
            except Exception as exc:
                LOG.exception("hybrid pose loop failed")
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
    server_version = "VQPoseHybrid/0.1"

    def log_message(self, fmt, *args):
        LOG.debug(fmt, *args)

    def do_GET(self):
        if self.path in {"/healthz", "/status"}:
            value = self.server.runtime.status()
            status = HTTPStatus.OK if self.path == "/status" or value["ready"] else HTTPStatus.SERVICE_UNAVAILABLE
            return self._json(status, value)
        if self.path == "/frame.jpg":
            frame = self.server.runtime.frame()
            if frame is None:
                return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "frame_unavailable"})
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return self.wfile.write(frame)
        if self.path == "/live.mjpg":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sequence = 0
            try:
                while True:
                    value = self.server.runtime.wait_frame(sequence)
                    if value is None:
                        continue
                    sequence, frame = value
                    self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

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
    parser.add_argument("--port", type=int, default=8792)
    parser.add_argument("--camera-url", default="http://127.0.0.1:8765/api/camera/live.mjpg")
    parser.add_argument("--worker-url", default="http://127.0.0.1:8793/infer")
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument("--sample-fps", type=float, default=10.0)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1"}:
        parser.error("pose service may bind only to loopback")
    logging.basicConfig(level=logging.INFO)
    runtime = PoseRuntime(args.camera_url, args.worker_url,
                          rotation=args.rotation, sample_fps=args.sample_fps)
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
