#!/usr/bin/env python3
"""Stateful BlazePose worker: sparse CPU detector + QNN/HTP landmarks."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from utils.tf import load_qnn_delegate

from pose_service.detector_geometry import decode_pose_detections, roi_from_detection
from pose_service.hybrid_geometry import SquareRoi, roi_from_auxiliary


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0)))


def refine_from_heatmap(landmarks: np.ndarray, heatmap: np.ndarray, kernel_size: int = 7):
    """Port of MediaPipe RefineLandmarksFromHeatmapCalculator defaults."""

    height, width, channels = heatmap.shape
    refined = landmarks.copy()
    offset = (kernel_size - 1) // 2
    for index in range(min(len(refined), channels)):
        center_col = int(refined[index, 0] / 256.0 * width)
        center_row = int(refined[index, 1] / 256.0 * height)
        if not (0 <= center_col < width and 0 <= center_row < height):
            continue
        begin_col, end_col = max(0, center_col - offset), min(width, center_col + offset + 1)
        begin_row, end_row = max(0, center_row - offset), min(height, center_row + offset + 1)
        area = sigmoid(heatmap[begin_row:end_row, begin_col:end_col, index])
        if area.size == 0 or float(area.max()) < 0.5:
            continue
        total = float(area.sum())
        rows, cols = np.mgrid[begin_row:end_row, begin_col:end_col]
        refined[index, 0] = float((cols * area).sum() / total / width * 256.0)
        refined[index, 1] = float((rows * area).sum() / total / height * 256.0)
    return refined


class Runtime:
    def __init__(self, landmark_model: str, detector_model: str, detector_refresh: float = 2.0):
        delegates = load_qnn_delegate(delegate_options={
            "backend_type": "htp",
            "htp_performance_mode": "2",
            "htp_precision": "1",
            "log_level": "1",
        })
        self.interpreter = Interpreter(landmark_model, experimental_delegates=delegates)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.outputs = {item["name"]: item for item in self.interpreter.get_output_details()}
        self.detector = Interpreter(detector_model)
        self.detector.allocate_tensors()
        self.detector_input = self.detector.get_input_details()[0]
        detector_outputs = self.detector.get_output_details()
        self.detector_boxes = next(item for item in detector_outputs if list(item["shape"])[-1] == 12)
        self.detector_scores = next(item for item in detector_outputs if list(item["shape"])[-1] == 1)
        self.detector_refresh = detector_refresh
        self.roi = None
        self.last_detection_at = 0.0
        self.misses = 0
        self.detector_runs = 0
        self.timings = deque(maxlen=300)
        self.detector_timings = deque(maxlen=100)
        self.lock = threading.Lock()

    @staticmethod
    def _letterbox(frame):
        height, width = frame.shape[:2]
        scale = min(224.0 / width, 224.0 / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        left = (224 - resized_width) // 2
        top = (224 - resized_height) // 2
        canvas = np.zeros((224, 224, 3), dtype=np.uint8)
        canvas[top:top + resized_height, left:left + resized_width] = resized
        padding = (left / 224.0, top / 224.0,
                   (224 - left - resized_width) / 224.0,
                   (224 - top - resized_height) / 224.0)
        return canvas, padding

    @staticmethod
    def _crop(frame, roi: SquareRoi):
        cosine, sine = math.cos(roi.rotation), math.sin(roi.rotation)
        step = roi.size / 256.0
        matrix = np.asarray((
            (cosine * step, -sine * step,
             roi.center_x + (-0.5 * cosine + 0.5 * sine) * roi.size),
            (sine * step, cosine * step,
             roi.center_y + (-0.5 * sine - 0.5 * cosine) * roi.size),
        ), dtype=np.float32)
        return cv2.warpAffine(frame, matrix, (256, 256),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT)

    def _detect(self, rgb, width, height):
        image, padding = self._letterbox(rgb)
        tensor = image.astype(np.float32)[None]
        if self.detector_input["dtype"] == np.float32:
            tensor = tensor / 127.5 - 1.0
        started = time.perf_counter()
        self.detector.set_tensor(self.detector_input["index"], tensor.astype(self.detector_input["dtype"]))
        self.detector.invoke()
        elapsed = (time.perf_counter() - started) * 1000.0
        self.detector_timings.append(elapsed)
        self.detector_runs += 1
        boxes = self.detector.get_tensor(self.detector_boxes["index"])[0]
        scores = self.detector.get_tensor(self.detector_scores["index"])[0]
        detections = decode_pose_detections(boxes, scores, padding, minimum_score=.35)
        return (roi_from_detection(detections[0], width, height) if detections else None,
                detections[0].score if detections else None, elapsed)

    @staticmethod
    def _project(records, roi, width, height):
        output = []
        for record in records:
            x, y = roi.project(float(record["x"]), float(record["y"]), width, height)
            output.append({
                "x": round(x, 7),
                "y": round(y, 7),
                "z": round(float(record.get("z", 0.0)) * roi.size / width, 7),
                "visibility": round(float(record.get("visibility", 0.0)), 7),
                "presence": round(float(record.get("presence", 0.0)), 7),
            })
        return output

    @staticmethod
    def _upper_body_valid(records, threshold=.5):
        return len(records) >= 13 and all(
            min(item["visibility"], item["presence"]) >= threshold
            and 0.0 <= item["x"] <= 1.0 and 0.0 <= item["y"] <= 1.0
            for item in (records[index] for index in (0, 7, 8, 11, 12))
        )

    def infer(self, encoded: bytes):
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("invalid_jpeg")
        with self.lock:
            started_total = time.perf_counter()
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.monotonic()
            detector_ran = self.roi is None or now - self.last_detection_at >= self.detector_refresh
            detector_score = None
            detector_ms = None
            if detector_ran:
                detected_roi, detector_score, detector_ms = self._detect(rgb, width, height)
                self.last_detection_at = now
                if detected_roi is not None:
                    self.roi = detected_roi
            if self.roi is None:
                return self._empty(detector_ran, detector_score, detector_ms,
                                   (time.perf_counter() - started_total) * 1000.0)
            active_roi = self.roi
            image = cv2.cvtColor(self._crop(frame, active_roi), cv2.COLOR_BGR2RGB)
            tensor = image.astype(np.float32)[None] / 255.0
            self.interpreter.set_tensor(self.input["index"], tensor)
            before = time.perf_counter()
            self.interpreter.invoke()
            elapsed = (time.perf_counter() - before) * 1000.0
            self.timings.append(elapsed)
            values = {
                name: self.interpreter.get_tensor(detail["index"]).copy()
                for name, detail in self.outputs.items()
            }
            landmarks = refine_from_heatmap(
                values["Identity"].reshape(39, 5), values["Identity_3"].reshape(64, 64, 39),
            )
            world = values["Identity_4"].reshape(39, 3)
            local_records = [{
                "x": float(item[0] / 256.0), "y": float(item[1] / 256.0),
                "z": float(item[2] / 256.0), "visibility": float(sigmoid(item[3])),
                "presence": float(sigmoid(item[4])),
            } for item in landmarks[:33]]
            records = self._project(local_records, active_roi, width, height)
            pose_presence = float(values["Identity_1"].reshape(-1)[0])
            accepted = pose_presence >= .5 or self._upper_body_valid(records)
            if accepted:
                self.misses = 0
                tracked = roi_from_auxiliary([
                    {"x": float(landmarks[index, 0] / 256.0),
                     "y": float(landmarks[index, 1] / 256.0)} for index in (33, 34)
                ], active_roi, width, height)
                if tracked is not None:
                    self.roi = tracked
            else:
                self.misses += 1
                if self.misses >= 3:
                    self.roi = None
            cosine, sine = math.cos(active_roi.rotation), math.sin(active_roi.rotation)
            world_records = [{
                "x": round(float(cosine * point[0] - sine * point[1]), 7),
                "y": round(float(sine * point[0] + cosine * point[1]), 7),
                "z": round(float(point[2]), 7),
            } for point in world[:33]]
            total_ms = (time.perf_counter() - started_total) * 1000.0
            return {
                "backend": "qnn-htp-fp16",
                "npu_accelerated": True,
                "pose_presence": pose_presence,
                "detected": accepted,
                "detector_ran": detector_ran,
                "detector_score": detector_score,
                "detector_ms": round(detector_ms, 3) if detector_ms is not None else None,
                "inference_ms": round(elapsed, 3),
                "total_ms": round(total_ms, 3),
                "landmarks": records,
                "world_landmarks": world_records,
            }

    def _empty(self, detector_ran, score, detector_ms, total_ms):
        return {
            "backend": "qnn-htp-fp16", "npu_accelerated": True, "detected": False,
            "pose_presence": 0.0, "detector_ran": detector_ran,
            "detector_score": score,
            "detector_ms": round(detector_ms, 3) if detector_ms is not None else None,
            "inference_ms": None, "total_ms": round(total_ms, 3),
            "landmarks": [], "world_landmarks": [],
        }

    def status(self):
        values = list(self.timings)
        detector_values = list(self.detector_timings)
        result = {
            "ready": True,
            "backend": "qnn-htp-fp16",
            "npu_accelerated": True,
            "samples": len(values),
            "latency_median_ms": round(statistics.median(values), 3) if values else None,
            "detector_runs": self.detector_runs,
            "detector_latency_median_ms": round(statistics.median(detector_values), 3)
            if detector_values else None,
            "detector_refresh_seconds": self.detector_refresh,
            "tracking": self.roi is not None,
        }
        if getattr(self,"shared_tts",None) is not None:
            result["shared_tts"]=self.shared_tts.status()
        return result


class Handler(BaseHTTPRequestHandler):
    server_version = "VQPoseQNN/0.1"

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path != "/healthz":
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        self._json(HTTPStatus.OK, self.server.runtime.status())

    def do_POST(self):
        if self.path == "/_internal/qnn-tts":
            from qnn_shared import handle_http
            return handle_http(self,getattr(self.server.runtime,"shared_tts",None))
        if self.path != "/infer":
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        length = int(self.headers.get("Content-Length", "0"))
        if not 1 <= length <= 1_500_000:
            return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "invalid_size"})
        try:
            result = self.server.runtime.infer(self.rfile.read(length))
            self._json(HTTPStatus.OK, result)
        except Exception as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": type(exc).__name__})

    def _json(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark-model", required=True)
    parser.add_argument("--detector-model", required=True)
    parser.add_argument("--detector-refresh", type=float, default=2.0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8793)
    parser.add_argument("--shared-tts-root")
    parser.add_argument("--shared-tts-library")
    parser.add_argument("--shared-tts-models")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.runtime = Runtime(args.landmark_model, args.detector_model, args.detector_refresh)
    if any((args.shared_tts_root,args.shared_tts_library,args.shared_tts_models)):
        if not all((args.shared_tts_root,args.shared_tts_library,args.shared_tts_models)):
            raise ValueError("all shared TTS paths must be configured")
        from qnn_shared import start_shared
        server.runtime.shared_tts=start_shared(args.shared_tts_root,args.shared_tts_models,
                                               args.shared_tts_library,server.runtime.lock)
    server.serve_forever()


if __name__ == "__main__":
    main()
