#!/usr/bin/env python3
"""Compare same-frame canonical results from CPU and NPU pose backends."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from pose_service.canonical import CanonicalPoint, CanonicalPose, posture_features


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def pose(record: dict, backend: str) -> CanonicalPose:
    points = {name: CanonicalPoint(**value) for name, value in record.get("points", {}).items()}
    return CanonicalPose(backend, points, record.get("inference_ms", 0), record["id"])


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * fraction)]


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--npu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    cpu, npu = load(args.cpu), load(args.npu)
    cpu_records = {item["id"]: item for item in cpu["records"]}
    npu_records = {item["id"]: item for item in npu["records"]}
    ids = sorted(set(cpu_records) & set(npu_records))
    coordinate_errors, feature_deltas = [], {}
    cpu_pose = npu_pose = cpu_upper = npu_upper = paired_upper = 0
    for frame_id in ids:
        left = pose(cpu_records[frame_id], cpu["backend"])
        right = pose(npu_records[frame_id], npu["backend"])
        cpu_pose += bool(left.points)
        npu_pose += bool(right.points)
        left_valid = left.upper_body_valid(args.threshold)
        right_valid = right.upper_body_valid(args.threshold)
        cpu_upper += left_valid
        npu_upper += right_valid
        if left_valid and right_valid:
            paired_upper += 1
            for name in set(left.points) & set(right.points):
                if left.points[name].score >= args.threshold and right.points[name].score >= args.threshold:
                    coordinate_errors.append(math.hypot(
                        left.points[name].x - right.points[name].x,
                        left.points[name].y - right.points[name].y,
                    ))
            left_features = posture_features(left, args.threshold)
            right_features = posture_features(right, args.threshold)
            if left_features and right_features:
                for name in set(left_features["values"]) & set(right_features["values"]):
                    feature_deltas.setdefault(name, []).append(abs(
                        left_features["values"][name] - right_features["values"][name]
                    ))
    count = len(ids)
    report = {
        "schema": "vq.pose-shadow-comparison.v1",
        "label": cpu.get("label"),
        "matched_frames": count,
        "threshold": args.threshold,
        "cpu": {
            "backend": cpu["backend"],
            "pose_rate": round(cpu_pose / count, 3) if count else 0,
            "upper_body_rate": round(cpu_upper / count, 3) if count else 0,
            "latency_ms": cpu["latency_ms"],
        },
        "npu": {
            "backend": npu["backend"],
            "pose_rate": round(npu_pose / count, 3) if count else 0,
            "upper_body_rate": round(npu_upper / count, 3) if count else 0,
            "latency_ms": npu["latency_ms"],
        },
        "paired_upper_body_frames": paired_upper,
        "coordinate_error_normalized": {
            "samples": len(coordinate_errors),
            "median": rounded(statistics.median(coordinate_errors)) if coordinate_errors else None,
            "p95": rounded(percentile(coordinate_errors, .95)),
        },
        "feature_absolute_delta_median": {
            name: rounded(statistics.median(values)) for name, values in sorted(feature_deltas.items())
        },
        "posture_state_agreement": None,
        "eligible_for_production": False,
        "note": "State agreement requires labelled normal/deviation segments and per-backend calibration.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
