"""Personal pose baseline and deterministic relative-deviation classifier."""

from __future__ import annotations

import statistics
from dataclasses import dataclass


CORE_FEATURES = (
    "head_forward_ratio", "head_height_ratio",
    "shoulder_slope_ratio",
)
MAX_CALIBRATION_MAD = {
    "head_forward_ratio": 0.05,
    "head_height_ratio": 0.08,
    "shoulder_slope_ratio": 0.12,
    "torso_lean_degrees": 5.0,
    "nose_ear_drop_ratio": 0.08,
}


@dataclass(frozen=True)
class PostureBaseline:
    values: dict[str, float]
    mad: dict[str, float]
    sample_count: int


@dataclass(frozen=True)
class PostureAssessment:
    visible: bool
    confidence: float
    reasons: tuple[str, ...]


def build_posture_baseline(samples, *, minimum_samples=20):
    """Use robust medians; reject short or structurally incomplete calibration."""
    usable = [sample for sample in samples if isinstance(sample, dict)
              and isinstance(sample.get("values"), dict)]
    if len(usable) < minimum_samples:
        raise ValueError("not enough visible calibration samples")
    columns = {}
    for key in CORE_FEATURES + ("shoulder_width", "torso_lean_degrees", "nose_ear_drop_ratio"):
        values = [float(sample["values"][key]) for sample in usable
                  if key in sample["values"]]
        if key in CORE_FEATURES and len(values) < minimum_samples:
            raise ValueError(f"missing calibration feature: {key}")
        if len(values) >= minimum_samples:
            median = statistics.median(values)
            mad = statistics.median(abs(value - median) for value in values)
            if key == "nose_ear_drop_ratio" and mad > MAX_CALIBRATION_MAD[key]:
                # Optional coverage must have its own stable observed baseline.
                # A moving nose must not fabricate a neutral value or invalidate
                # the established legacy head/shoulder calibration.
                continue
            columns[key] = (median, mad)
    medians = {key: round(value[0], 5) for key, value in columns.items()}
    deviations = {key: round(value[1], 5) for key, value in columns.items()}
    unstable = [key for key, limit in MAX_CALIBRATION_MAD.items()
                if key in deviations and deviations[key] > limit]
    if unstable:
        raise ValueError("unstable calibration features: " + ",".join(unstable))
    return PostureBaseline(
        values=medians,
        mad=deviations,
        sample_count=len(usable),
    )


def assess_posture(features, baseline):
    """Classify only meaningful change from the user's own healthy baseline."""
    if not isinstance(features, dict) or not isinstance(features.get("values"), dict):
        return PostureAssessment(False, 0.0, ())
    values = features["values"]
    if any(key not in values for key in CORE_FEATURES):
        return PostureAssessment(False, 0.0, ())
    confidence = max(0.0, min(1.0, float(features.get("confidence", 0.0))))
    if confidence < 0.5:
        return PostureAssessment(False, confidence, ())

    reasons = []
    head_forward_limit = max(0.16, 5 * baseline.mad["head_forward_ratio"])
    if abs(values["head_forward_ratio"] - baseline.values["head_forward_ratio"]) > head_forward_limit:
        reasons.append("head_forward")

    head_drop_limit = max(0.13, 5 * baseline.mad["head_height_ratio"])
    if baseline.values["head_height_ratio"] - values["head_height_ratio"] > head_drop_limit:
        if "head_forward" not in reasons:
            reasons.append("head_forward")

    # Old persisted baselines remain valid. This additional low-head cue is
    # enabled only after real calibration supplied both its median and MAD.
    if ("nose_ear_drop_ratio" in baseline.values and "nose_ear_drop_ratio" in baseline.mad
            and "nose_ear_drop_ratio" in values):
        nose_drop_limit = max(0.13, 5 * baseline.mad["nose_ear_drop_ratio"])
        if values["nose_ear_drop_ratio"] - baseline.values["nose_ear_drop_ratio"] > nose_drop_limit:
            if "head_forward" not in reasons:
                reasons.append("head_forward")

    shoulder_limit = max(0.10, 5 * baseline.mad["shoulder_slope_ratio"])
    if values["shoulder_slope_ratio"] - baseline.values["shoulder_slope_ratio"] > shoulder_limit:
        reasons.append("shoulder_asymmetry")

    if ("torso_lean_degrees" in baseline.values
            and "torso_lean_degrees" in values):
        torso_limit = max(8.0, 5 * baseline.mad["torso_lean_degrees"])
        if values["torso_lean_degrees"] - baseline.values["torso_lean_degrees"] > torso_limit:
            reasons.append("torso_lean")

    return PostureAssessment(True, confidence, tuple(sorted(set(reasons))))
