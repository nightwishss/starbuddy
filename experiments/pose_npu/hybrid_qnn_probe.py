#!/usr/bin/env python3
"""Run the stateful CPU-detector/QNN-landmark worker over shadow frames."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

import cv2
from pose_service.canonical import MEDIAPIPE_INDEX


def invoke(worker_url, image):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("crop encode failed")
    before = time.perf_counter()
    request = urllib.request.Request(worker_url, data=encoded.tobytes(), method="POST",
                                     headers={"Content-Type": "image/jpeg"})
    with urllib.request.urlopen(request, timeout=3) as response:
        result = json.load(response)
    return result, (time.perf_counter() - before) * 1000.0


def point_json(point):
    return {
        "x": max(0.0, min(1.0, float(point["x"]))),
        "y": max(0.0, min(1.0, float(point["y"]))),
        "z": float(point["z"]),
        "score": max(0.0, min(1.0, min(float(point["visibility"]), float(point["presence"])))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-url", default="http://127.0.0.1:8793/infer")
    args = parser.parse_args()
    manifest = json.loads((args.frames_dir / "manifest.json").read_text())
    records, timings = [], []
    interval = 1.0 / float(manifest.get("fps_target", 5.0))
    next_frame_at = time.monotonic()
    for name in manifest["frames"]:
        delay = next_frame_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        next_frame_at += interval
        frame = cv2.imread(str(args.frames_dir / name))
        if frame is None:
            raise RuntimeError(f"cannot read {name}")
        response, elapsed = invoke(args.worker_url, frame)
        timings.append(elapsed)
        record = {"id": name, "inference_ms": round(elapsed, 3), "points": {}}
        if response.get("detected"):
            record["points"] = {
                name_: point_json(response["landmarks"][position])
                for name_, position in MEDIAPIPE_INDEX.items()
            }
        records.append(record)
    ordered = sorted(timings[1:] or timings)
    report = {
        "schema": "vq.pose-shadow-backend.v1",
        "backend": "blazepose-cpu-detector-qnn-htp-landmark-fp16",
        "label": manifest["label"],
        "frame_count": len(records),
        "replay_mode": "source-paced",
        "raw_frames_retained_by_probe": False,
        "latency_ms": {
            "first": round(timings[0], 3) if timings else None,
            "warm_median": round(statistics.median(ordered), 3) if ordered else None,
            "warm_p95": round(ordered[int((len(ordered) - 1) * .95)], 3) if ordered else None,
        },
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
