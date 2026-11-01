"""Geometry shared by the CPU-bootstrap/QNN-landmark pose backend."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace


@dataclass(frozen=True)
class SquareRoi:
    """Square image ROI expressed in source-image pixels."""

    center_x: float
    center_y: float
    size: float
    rotation: float = 0.0

    def project(self, x: float, y: float, image_width: int, image_height: int) -> tuple[float, float]:
        local_x = (x - 0.5) * self.size
        local_y = (y - 0.5) * self.size
        cosine, sine = math.cos(self.rotation), math.sin(self.rotation)
        return (
            (self.center_x + cosine * local_x - sine * local_y) / image_width,
            (self.center_y + sine * local_x + cosine * local_y) / image_height,
        )


def roi_from_landmarks(landmarks, image_width: int, image_height: int, *, scale: float = 1.45):
    """Build a stable square ROI from MediaPipe landmarks.

    This is only the bootstrap ROI. Once the QNN model produces its two
    auxiliary alignment points, ``roi_from_auxiliary`` replaces it with the
    same feedback geometry used by the official MediaPipe graph.
    """

    usable = [
        point for point in landmarks
        if float(getattr(point, "visibility", 0.0) or 0.0) >= 0.35
        and float(getattr(point, "presence", 0.0) or 0.0) >= 0.35
        and 0.0 <= float(point.x) <= 1.0
        and 0.0 <= float(point.y) <= 1.0
    ]
    if len(usable) < 5:
        return None
    xs = [float(point.x) * image_width for point in usable]
    ys = [float(point.y) * image_height for point in usable]
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    size = max(max(xs) - min(xs), max(ys) - min(ys)) * scale
    size = max(size, min(image_width, image_height) * 0.32)
    size = min(size, max(image_width, image_height) * 1.5)
    return SquareRoi(center_x, center_y, size)


def roi_from_auxiliary(records, parent: SquareRoi, image_width: int, image_height: int):
    """Recreate AlignmentPointsRects + 1.25 RectTransformation geometry."""

    if len(records) < 2:
        return None
    center_norm = parent.project(float(records[0]["x"]), float(records[0]["y"]),
                                 image_width, image_height)
    scale_norm = parent.project(float(records[1]["x"]), float(records[1]["y"]),
                                image_width, image_height)
    center_x, center_y = center_norm[0] * image_width, center_norm[1] * image_height
    scale_x, scale_y = scale_norm[0] * image_width, scale_norm[1] * image_height
    dx, dy = scale_x - center_x, scale_y - center_y
    size = math.hypot(dx, dy) * 2.0 * 1.25
    if not math.isfinite(size) or size < min(image_width, image_height) * 0.2:
        return None
    rotation = math.pi / 2.0 - math.atan2(-dy, dx)
    rotation = rotation - 2.0 * math.pi * math.floor((rotation + math.pi) / (2.0 * math.pi))
    return SquareRoi(center_x, center_y, size, rotation)


def project_worker_landmarks(records, roi: SquareRoi, image_width: int, image_height: int):
    """Convert QNN worker ROI-relative dictionaries into metric-compatible points."""

    projected = []
    for record in records:
        x, y = roi.project(float(record["x"]), float(record["y"]), image_width, image_height)
        projected.append(SimpleNamespace(
            x=x,
            y=y,
            z=float(record.get("z", 0.0)) * roi.size / image_width,
            visibility=float(record.get("visibility", 0.0)),
            presence=float(record.get("presence", 0.0)),
        ))
    return projected


def upper_body_worker_valid(landmarks, threshold: float = 0.5) -> bool:
    """Accept a QNN result when all posture-critical upper points are usable.

    The standalone landmark model's scalar pose flag is sensitive to partial
    body framing. A CPU bootstrap already established that this ROI contains a
    person, so tracked-frame validity is based on the five points consumed by
    posture policy rather than that scalar alone.
    """

    required = (0, 7, 8, 11, 12)
    return len(landmarks) > max(required) and all(
        min(float(landmarks[index].visibility), float(landmarks[index].presence)) >= threshold
        and 0.0 <= float(landmarks[index].x) <= 1.0
        and 0.0 <= float(landmarks[index].y) <= 1.0
        for index in required
    )
