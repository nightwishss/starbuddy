#!/usr/bin/env python3
"""Run the frozen MediaPipe CPU model over a bounded frame directory."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pose_service.canonical import MEDIAPIPE_INDEX, from_mediapipe, posture_features


def point_json(point) -> dict:
    return {"x": point.x, "y": point.y, "z": point.z, "score": point.score}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.frames_dir / "manifest.json").read_text())
    frame_interval_ms = round(1000 / float(manifest.get("fps_target", 5.0)))
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    records, timings = [], []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for index, name in enumerate(manifest["frames"]):
            frame = cv2.imread(str(args.frames_dir / name))
            if frame is None:
                raise RuntimeError(f"cannot read {name}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            before = time.perf_counter()
            result = landmarker.detect_for_video(image, index * frame_interval_ms)
            elapsed = (time.perf_counter() - before) * 1000
            timings.append(elapsed)
            record = {"id": name, "inference_ms": round(elapsed, 3), "points": {}}
            if result.pose_landmarks:
                pose = from_mediapipe(result.pose_landmarks[0], elapsed, name)
                record["points"] = {key: point_json(value) for key, value in pose.points.items()}
                record["features"] = posture_features(pose, threshold=0.5)
            records.append(record)
    ordered = sorted(timings[1:] or timings)
    report = {
        "schema": "vq.pose-shadow-backend.v1",
        "backend": "mediapipe-cpu-xnnpack",
        "label": manifest["label"],
        "frame_count": len(records),
        "raw_frames_retained_by_probe": False,
        "latency_ms": {
            "first": round(timings[0], 3),
            "warm_median": round(statistics.median(ordered), 3),
            "warm_p95": round(ordered[int((len(ordered) - 1) * .95)], 3),
        },
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
