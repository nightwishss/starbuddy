"""Pure landmark-quality helpers shared by the pose service and tests."""

import math

UPPER_BODY_LANDMARKS = (0, 7, 8, 11, 12)
FULL_CORE_LANDMARKS = UPPER_BODY_LANDMARKS + (23, 24)


def landmark_valid(point, threshold=0.5):
    """Return true only for visible, present landmarks inside the image."""
    visibility = float(getattr(point, "visibility", 0.0) or 0.0)
    presence = float(getattr(point, "presence", 0.0) or 0.0)
    return (
        visibility >= threshold and presence >= threshold
        and 0.0 <= float(point.x) <= 1.0
        and 0.0 <= float(point.y) <= 1.0
    )


def landmark_quality(landmarks, threshold=0.5):
    """Summarize framing quality without exposing landmark coordinates."""
    if not landmarks or len(landmarks) < 25:
        return {"upper_body_valid": False, "full_core_valid": False,
                "upper_body_visibility": 0.0}
    upper_valid = all(landmark_valid(landmarks[i], threshold) for i in UPPER_BODY_LANDMARKS)
    full_valid = all(landmark_valid(landmarks[i], threshold) for i in FULL_CORE_LANDMARKS)
    visibility = sum(float(landmarks[i].visibility or 0.0) for i in UPPER_BODY_LANDMARKS)
    return {
        "upper_body_valid": upper_valid,
        "full_core_valid": full_valid,
        "upper_body_visibility": round(visibility / len(UPPER_BODY_LANDMARKS), 3),
    }


def posture_features(landmarks, world_landmarks=None, threshold=0.5):
    """Build camera-relative ergonomic features without exposing coordinates."""
    quality = landmark_quality(landmarks, threshold)
    if not quality["upper_body_valid"]:
        return None
    left_ear, right_ear = landmarks[7], landmarks[8]
    left_shoulder, right_shoulder = landmarks[11], landmarks[12]
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
    values = {"shoulder_width": shoulder_width}
    if world_landmarks and len(world_landmarks) >= 25:
        world_nose = world_landmarks[0]
        we_left, we_right = world_landmarks[7], world_landmarks[8]
        ws_left, ws_right = world_landmarks[11], world_landmarks[12]
        we = tuple((getattr(we_left, axis) + getattr(we_right, axis)) / 2.0
                   for axis in ("x", "y", "z"))
        ws = tuple((getattr(ws_left, axis) + getattr(ws_right, axis)) / 2.0
                   for axis in ("x", "y", "z"))
        world_shoulder_width = math.sqrt(sum(
            (getattr(ws_right, axis) - getattr(ws_left, axis)) ** 2
            for axis in ("x", "y", "z")
        ))
        if world_shoulder_width < 0.03:
            return None
        values.update({
            "head_forward_ratio": math.hypot(we[0] - ws[0], we[2] - ws[2]) / world_shoulder_width,
            "head_height_ratio": abs(ws[1] - we[1]) / world_shoulder_width,
            "nose_ear_drop_ratio": (world_nose.y - we[1]) / world_shoulder_width,
            "shoulder_slope_ratio": abs(ws_right.y - ws_left.y) / world_shoulder_width,
        })
    else:
        values.update({
            "head_forward_ratio": abs(ear_x - shoulder_x) / shoulder_width,
            "head_height_ratio": abs(shoulder_y - ear_y) / shoulder_width,
            "nose_ear_drop_ratio": (landmarks[0].y - ear_y) / shoulder_width,
            "shoulder_slope_ratio": abs(right_shoulder.y - left_shoulder.y) / shoulder_width,
        })
    if quality["full_core_valid"]:
        torso_points = world_landmarks if world_landmarks else landmarks
        left_hip, right_hip = torso_points[23], torso_points[24]
        torso_left_shoulder, torso_right_shoulder = torso_points[11], torso_points[12]
        hip_x = (left_hip.x + right_hip.x) / 2.0
        hip_y = (left_hip.y + right_hip.y) / 2.0
        hip_z = (getattr(left_hip, "z", 0.0) + getattr(right_hip, "z", 0.0)) / 2.0
        torso_shoulder_x = (torso_left_shoulder.x + torso_right_shoulder.x) / 2.0
        torso_shoulder_y = (torso_left_shoulder.y + torso_right_shoulder.y) / 2.0
        torso_shoulder_z = (
            getattr(torso_left_shoulder, "z", 0.0)
            + getattr(torso_right_shoulder, "z", 0.0)
        ) / 2.0
        vertical = abs(hip_y - torso_shoulder_y)
        values["torso_lean_degrees"] = math.degrees(
            math.atan2(math.hypot(torso_shoulder_x - hip_x, torso_shoulder_z - hip_z),
                       max(vertical, 1e-6))
        )
    confidence = min(
        min(float(landmarks[i].visibility or 0.0), float(landmarks[i].presence or 0.0))
        for i in UPPER_BODY_LANDMARKS
    )
    return {
        "values": {key: round(float(value), 5) for key, value in values.items()},
        "confidence": round(confidence, 3),
        "full_core_valid": quality["full_core_valid"],
    }
