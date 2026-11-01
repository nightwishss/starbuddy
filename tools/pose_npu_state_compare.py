#!/usr/bin/env python3
"""Compare personalized posture states from same-frame CPU/NPU reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from perception.posture import PostureSignal, PostureTemporalPolicy
from perception.posture_calibration import assess_posture, build_posture_baseline
from pose_service.canonical import CanonicalPoint, CanonicalPose, posture_features


def load(path):
    return json.loads(Path(path).read_text())


def features(record, backend):
    points = {name: CanonicalPoint(**point) for name, point in record.get("points", {}).items()}
    return posture_features(CanonicalPose(backend, points, record.get("inference_ms", 0), record["id"]), .5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--npu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, cpu, npu = load(args.manifest), load(args.cpu), load(args.npu)
    labels = manifest["frame_labels"]
    elapsed = manifest["frame_elapsed_seconds"]
    reports = {"cpu": cpu, "npu": npu}
    baselines, states, temporal = {}, {}, {}
    for key, report in reports.items():
        records = {item["id"]: item for item in report["records"]}
        baseline_samples = [features(records[name], report["backend"])
                            for name in manifest["frames"] if labels[name] == "healthy_baseline"]
        baseline_samples = [item for item in baseline_samples if item]
        baselines[key] = build_posture_baseline(baseline_samples)
        policy = PostureTemporalPolicy()
        states[key], temporal[key] = {}, {}
        for name in manifest["frames"]:
            if labels[name] == "healthy_baseline":
                # The product collects a baseline without running its reminder
                # policy. Treating calibration noise as live deviations can
                # seed a false suspect state before monitoring has begun.
                states[key][name] = "calibrating"
                temporal[key][name] = "calibrating"
                continue
            item = features(records[name], report["backend"])
            assessment = assess_posture(item, baselines[key])
            state = "unknown" if not assessment.visible else ("deviation" if assessment.reasons else "healthy")
            states[key][name] = state
            decision = policy.update(PostureSignal(float(elapsed[name]), assessment.visible,
                                                    assessment.confidence, assessment.reasons))
            temporal[key][name] = decision.state

    names = manifest["frames"]
    direct_agree = sum(states["cpu"][name] == states["npu"][name] for name in names)
    temporal_agree = sum(temporal["cpu"][name] == temporal["npu"][name] for name in names)
    segments = defaultdict(lambda: {"frames": 0, "cpu": Counter(), "npu": Counter(),
                                    "cpu_temporal": Counter(), "npu_temporal": Counter()})
    for name in names:
        segment = segments[labels[name]]
        segment["frames"] += 1
        for key in ("cpu", "npu"):
            segment[key][states[key][name]] += 1
            segment[key + "_temporal"][temporal[key][name]] += 1
    report = {
        "schema": "vq.pose-state-agreement.v1",
        "label": manifest["label"],
        "frames": len(names),
        "direct_state_agreement": round(direct_agree / len(names), 4),
        "temporal_state_agreement": round(temporal_agree / len(names), 4),
        "baselines": {key: {"values": value.values, "mad": value.mad,
                             "sample_count": value.sample_count}
                      for key, value in baselines.items()},
        "segments": {name: {**value,
                             "cpu": dict(value["cpu"]), "npu": dict(value["npu"]),
                             "cpu_temporal": dict(value["cpu_temporal"]),
                             "npu_temporal": dict(value["npu_temporal"])}
                     for name, value in segments.items()},
        "eligible_state_agreement": temporal_agree / len(names) >= .95,
        "calibration_excluded_from_monitoring": True,
        "raw_frames_retained": False,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
