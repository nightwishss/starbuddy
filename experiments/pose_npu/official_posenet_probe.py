#!/usr/bin/env python3
"""Bounded end-to-end smoke probe for Arduino's official PoseNet QNN image."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def upper_body_valid(person: dict, threshold: float = 0.1) -> bool:
    wanted = {"nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder"}
    scores = {
        item.get("name"): float(item.get("score", 0.0))
        for item in person.get("keypoints", [])
    }
    return all(scores.get(name, 0.0) >= threshold for name in wanted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url")
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--center-portrait-crop", action="store_true",
                        help="crop a centered 257:513 ROI before inference")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rotation", type=int, choices=(0, 180), default=0)
    parser.add_argument("--min-person-score", type=float, default=0.25)
    args = parser.parse_args()
    if not 2 <= args.iterations <= 300:
        parser.error("iterations must be between 2 and 300")
    if args.source_url and args.frames_dir:
        parser.error("source-url and frames-dir are mutually exclusive")

    os.chdir("/app")
    sys.path.insert(0, "/app")
    started = time.perf_counter()
    import inference  # noqa: E402  (model/delegate initialization is measured)
    inference.apply_config({"min_person_score": args.min_person_score})

    load_ms = (time.perf_counter() - started) * 1000
    capture = cv2.VideoCapture(args.source_url) if args.source_url else None
    if capture is not None and not capture.isOpened():
        raise SystemExit("camera stream unavailable")
    manifest = None
    if args.frames_dir:
        manifest = json.loads((args.frames_dir / "manifest.json").read_text())
        frame_names = manifest["frames"][:args.iterations]
    else:
        frame_names = [None] * args.iterations

    timings = []
    detected_frames = 0
    upper_body_frames = 0
    person_scores = []
    upper_min_scores = []
    keypoint_scores_by_name = {}
    records = []
    metadata = None
    try:
        for index, frame_name in enumerate(frame_names):
            if args.frames_dir:
                frame = cv2.imread(str(args.frames_dir / frame_name))
                if frame is None:
                    raise RuntimeError(f"cannot read {frame_name}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif capture is None:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            else:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("camera stream disconnected")
                if args.rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            full_height, full_width = frame.shape[:2]
            crop_x = 0
            if args.center_portrait_crop:
                crop_width = min(full_width, max(1, round(full_height * 257 / 513)))
                crop_x = (full_width - crop_width) // 2
                frame = np.ascontiguousarray(frame[:, crop_x:crop_x + crop_width])
            before = time.perf_counter()
            _, metadata = inference.inference_callback(frame.copy())
            timings.append((time.perf_counter() - before) * 1000)
            people = metadata.get("persons", []) if metadata else []
            record = {
                "id": frame_name or f"frame-{index:04d}",
                "inference_ms": round(timings[-1], 3),
                "points": {},
            }
            if people:
                detected_frames += 1
                person_scores.append(float(people[0].get("score", 0.0)))
                wanted = {"nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder"}
                for item in people[0].get("keypoints", []):
                    keypoint_scores_by_name.setdefault(item.get("name"), []).append(
                        float(item.get("score", 0.0))
                    )
                scores = [
                    float(item.get("score", 0.0))
                    for item in people[0].get("keypoints", [])
                    if item.get("name") in wanted
                ]
                if len(scores) == len(wanted):
                    upper_min_scores.append(min(scores))
                if upper_body_valid(people[0]):
                    upper_body_frames += 1
                height, width = frame.shape[:2]
                record["points"] = {
                    item["name"]: {
                        "x": max(0.0, min(1.0, (float(item["x"]) + crop_x) / full_width)),
                        "y": max(0.0, min(1.0, float(item["y"]) / full_height)),
                        "z": 0.0,
                        "score": max(0.0, min(1.0, float(item.get("score", 0.0)))),
                    }
                    for item in people[0].get("keypoints", [])
                    if item.get("name") in {
                        "nose", "left_ear", "right_ear", "left_shoulder",
                        "right_shoulder", "left_hip", "right_hip",
                    }
                }
            records.append(record)
    finally:
        if capture is not None:
            capture.release()

    warm = sorted(timings[1:])
    report = {
        "schema": "vq.official-posenet-qnn-smoke.v1",
        "backend": "posenet-litert-qnn-htp-w8a8",
        "source": "frame-directory" if args.frames_dir else (
            "camera-stream" if args.source_url else "synthetic-black-frame"
        ),
        "label": manifest.get("label") if manifest else None,
        "center_portrait_crop": args.center_portrait_crop,
        "raw_frames_retained": False,
        "model_load_ms": round(load_ms, 3),
        "iterations": len(timings),
        "first_frame_ms": round(timings[0], 3),
        "warm_median_ms": round(statistics.median(warm), 3),
        "warm_p95_ms": round(warm[int((len(warm) - 1) * 0.95)], 3),
        "warm_min_ms": round(min(warm), 3),
        "warm_max_ms": round(max(warm), 3),
        "latency_ms": {
            "first": round(timings[0], 3),
            "warm_median": round(statistics.median(warm), 3),
            "warm_p95": round(warm[int((len(warm) - 1) * 0.95)], 3),
        },
        "pose_success_rate": round(detected_frames / len(timings), 3),
        "upper_body_success_rate": round(upper_body_frames / len(timings), 3),
        "min_person_score": args.min_person_score,
        "detected_person_score": {
            "min": round(min(person_scores), 4) if person_scores else None,
            "median": round(statistics.median(person_scores), 4) if person_scores else None,
            "max": round(max(person_scores), 4) if person_scores else None,
        },
        "required_upper_keypoint_min_score": {
            "min": round(min(upper_min_scores), 4) if upper_min_scores else None,
            "median": round(statistics.median(upper_min_scores), 4) if upper_min_scores else None,
            "max": round(max(upper_min_scores), 4) if upper_min_scores else None,
        },
        "keypoint_score_median": {
            name: round(statistics.median(values), 4)
            for name, values in sorted(keypoint_scores_by_name.items())
        },
        "metadata_keys": sorted(metadata.keys()) if metadata else [],
        "records": records,
    }
    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
