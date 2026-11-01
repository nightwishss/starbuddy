"""Backend-neutral pose contract for posture experiments.

This module is intentionally not imported by the production MediaPipe runner
until an NPU backend has passed shadow acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


REQUIRED_UPPER = ("nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder")
OPTIONAL_CORE = ("left_hip", "right_hip")

MEDIAPIPE_INDEX = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
}


@dataclass(frozen=True)
class CanonicalPoint:
    x: float
    y: float
    score: float
    z: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x", "y", "z", "score"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.x <= 1.0 or not 0.0 <= self.y <= 1.0:
            raise ValueError("x and y must use normalized image coordinates")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between zero and one")


@dataclass(frozen=True)
class CanonicalPose:
    backend: str
    points: Mapping[str, CanonicalPoint]
    inference_ms: float
    frame_id: str | None = None

    def upper_body_valid(self, threshold: float = 0.1) -> bool:
        return all(name in self.points and self.points[name].score >= threshold for name in REQUIRED_UPPER)

    def full_core_valid(self, threshold: float = 0.1) -> bool:
        return self.upper_body_valid(threshold) and all(
            name in self.points and self.points[name].score >= threshold for name in OPTIONAL_CORE
        )


def from_mediapipe(
    landmarks: Sequence,
    inference_ms: float,
    frame_id: str | None = None,
) -> CanonicalPose:
    """Adapt MediaPipe's indexed landmarks without changing production code."""
    points = {}
    for name, index in MEDIAPIPE_INDEX.items():
        point = landmarks[index]
        visibility = float(getattr(point, "visibility", 0.0) or 0.0)
        presence = float(getattr(point, "presence", 0.0) or 0.0)
        points[name] = CanonicalPoint(
            float(point.x),
            float(point.y),
            min(visibility, presence),
            float(getattr(point, "z", 0.0) or 0.0),
        )
    return CanonicalPose("mediapipe-cpu", points, float(inference_ms), frame_id)


def from_posenet(
    keypoints: Mapping[str, object],
    inference_ms: float,
    frame_id: str | None = None,
) -> CanonicalPose:
    """Adapt PoseNet named points represented as objects or dictionaries."""
    def value(point: object, name: str, default: float = 0.0) -> float:
        if isinstance(point, Mapping):
            return float(point.get(name, default))
        return float(getattr(point, name, default))

    points = {
        name: CanonicalPoint(
            value(point, "x"), value(point, "y"),
            value(point, "score"), value(point, "z"),
        )
        for name, point in keypoints.items()
        if name in MEDIAPIPE_INDEX
    }
    return CanonicalPose("posenet-qnn", points, float(inference_ms), frame_id)


def posture_features(pose: CanonicalPose, threshold: float = 0.1) -> dict | None:
    """Compute the current image-relative posture features from named points."""
    if not pose.upper_body_valid(threshold):
        return None
    left_ear, right_ear = pose.points["left_ear"], pose.points["right_ear"]
    left_shoulder = pose.points["left_shoulder"]
    right_shoulder = pose.points["right_shoulder"]
    ear_x = (left_ear.x + right_ear.x) / 2.0
    ear_y = (left_ear.y + right_ear.y) / 2.0
    shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
    shoulder_width = math.hypot(
        right_shoulder.x - left_shoulder.x,
        right_shoulder.y - left_shoulder.y,
    )
    if shoulder_width < 0.03:
        return None
    values = {
        "shoulder_width": shoulder_width,
        "head_forward_ratio": abs(ear_x - shoulder_x) / shoulder_width,
        "head_height_ratio": abs(shoulder_y - ear_y) / shoulder_width,
        "shoulder_slope_ratio": abs(right_shoulder.y - left_shoulder.y) / shoulder_width,
    }
    full_core = pose.full_core_valid(threshold)
    if full_core:
        left_hip, right_hip = pose.points["left_hip"], pose.points["right_hip"]
        hip_x = (left_hip.x + right_hip.x) / 2.0
        hip_y = (left_hip.y + right_hip.y) / 2.0
        hip_z = (left_hip.z + right_hip.z) / 2.0
        shoulder_z = (left_shoulder.z + right_shoulder.z) / 2.0
        values["torso_lean_degrees"] = math.degrees(math.atan2(
            math.hypot(shoulder_x - hip_x, shoulder_z - hip_z),
            max(abs(hip_y - shoulder_y), 1e-6),
        ))
    confidence = min(pose.points[name].score for name in REQUIRED_UPPER)
    return {
        "values": {name: round(float(value), 5) for name, value in values.items()},
        "confidence": round(confidence, 3),
        "full_core_valid": full_core,
    }
