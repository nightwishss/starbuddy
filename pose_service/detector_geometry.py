"""Pure geometry for the standalone MediaPipe BlazePose detector.

The detector model works on a letterboxed 224x224 image and emits 2254 SSD
boxes.  Keeping this module independent from TensorFlow Lite makes the
decode/ROI contract unit-testable on the development host.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from pose_service.hybrid_geometry import SquareRoi


@dataclass(frozen=True)
class Detection:
    score: float
    bbox: tuple[float, float, float, float]
    keypoints: tuple[tuple[float, float], ...]


def pose_anchors():
    """Generate the fixed-size SSD anchors used by pose_detector.tflite."""

    anchors = []
    strides = (8, 16, 32, 32, 32)
    layer = 0
    while layer < len(strides):
        last = layer
        while last < len(strides) and strides[last] == strides[layer]:
            last += 1
        feature = math.ceil(224 / strides[layer])
        anchors_per_cell = 2 * (last - layer)
        for row in range(feature):
            for column in range(feature):
                center = ((column + 0.5) / feature, (row + 0.5) / feature)
                anchors.extend([center] * anchors_per_cell)
        layer = last
    if len(anchors) != 2254:
        raise RuntimeError(f"unexpected_pose_anchor_count:{len(anchors)}")
    return tuple(anchors)


def sigmoid(value: float) -> float:
    value = max(-100.0, min(100.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def remove_letterbox(value, padding):
    """Map a point from detector-square coordinates to source coordinates."""

    x, y = value
    left, top, right, bottom = padding
    return ((x - left) / (1.0 - left - right),
            (y - top) / (1.0 - top - bottom))


def decode_pose_detections(raw_boxes, raw_scores, padding=(0.0, 0.0, 0.0, 0.0),
                           *, minimum_score=0.35):
    """Decode detector tensors and return score-sorted source-image detections."""

    anchors = pose_anchors()
    detections = []
    for index, anchor in enumerate(anchors):
        score = sigmoid(raw_scores[index][0] if hasattr(raw_scores[index], "__len__")
                        else raw_scores[index])
        if score < minimum_score:
            continue
        record = raw_boxes[index]
        center = remove_letterbox(
            (float(record[0]) / 224.0 + anchor[0],
             float(record[1]) / 224.0 + anchor[1]), padding,
        )
        width = float(record[2]) / 224.0 / (1.0 - padding[0] - padding[2])
        height = float(record[3]) / 224.0 / (1.0 - padding[1] - padding[3])
        keypoints = tuple(remove_letterbox(
            (float(record[4 + 2 * keypoint]) / 224.0 + anchor[0],
             float(record[5 + 2 * keypoint]) / 224.0 + anchor[1]), padding,
        ) for keypoint in range(4))
        detections.append(Detection(
            score, (center[0] - width / 2.0, center[1] - height / 2.0, width, height),
            keypoints,
        ))
    return weighted_non_max_suppression(detections)


def _iou(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0.0 else 0.0


def weighted_non_max_suppression(detections, threshold=0.3):
    """MediaPipe-style weighted suppression for the normally single pose."""

    remaining = sorted(detections, key=lambda item: item.score, reverse=True)
    output = []
    while remaining:
        reference = remaining[0]
        overlap = [item for item in remaining if _iou(reference.bbox, item.bbox) > threshold]
        remaining = [item for item in remaining if item not in overlap]
        total = sum(item.score for item in overlap)
        bbox = tuple(sum(item.bbox[i] * item.score for item in overlap) / total for i in range(4))
        keypoints = tuple(tuple(
            sum(item.keypoints[keypoint][axis] * item.score for item in overlap) / total
            for axis in range(2)
        ) for keypoint in range(4))
        output.append(Detection(reference.score, bbox, keypoints))
    return output


def roi_from_detection(detection: Detection, image_width: int, image_height: int):
    """Build the official alignment-point ROI from detector keypoints 0 and 1."""

    if len(detection.keypoints) < 2:
        return None
    center = detection.keypoints[0]
    scale = detection.keypoints[1]
    center_x, center_y = center[0] * image_width, center[1] * image_height
    scale_x, scale_y = scale[0] * image_width, scale[1] * image_height
    dx, dy = scale_x - center_x, scale_y - center_y
    size = 2.0 * math.hypot(dx, dy) * 1.25
    if not math.isfinite(size) or size < 1.0:
        return None
    rotation = math.pi / 2.0 - math.atan2(-dy, dx)
    rotation -= 2.0 * math.pi * math.floor((rotation + math.pi) / (2.0 * math.pi))
    return SquareRoi(center_x, center_y, size, rotation)
