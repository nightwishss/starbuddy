"""Low-rate FER+ inference and conservative temporal cue policy."""

from __future__ import annotations

import collections
import math
import time


LABELS = ("neutral", "happy_like", "surprised_like", "sad_like",
          "angry_like", "disgust_like", "fear_like", "contempt_like")
MODEL_ID = "emotion-ferplus-8"
ADMISSION_OWNER = "model:affect-ferplus"
ADMISSION_RESOURCES = {"cpu_compute": 10}


def _softmax(scores):
    values = [float(value) for value in scores]
    if len(values) != len(LABELS) or not all(math.isfinite(value) for value in values):
        raise ValueError("invalid affect model output")
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    total = sum(exponents)
    return {label: round(value / total, 4) for label, value in zip(LABELS, exponents)}


class FerPlusModel:
    def __init__(self, path):
        import cv2
        cv2.setNumThreads(1)
        self.cv2 = cv2
        self.net = cv2.dnn.readNetFromONNX(str(path))

    def infer(self, jpeg, bbox, *, rotation=180):
        import numpy as np
        cv2 = self.cv2
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("invalid camera frame")
        rotations = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                     270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        if rotation in rotations:
            image = cv2.rotate(image, rotations[rotation])
        elif rotation != 0:
            raise ValueError("unsupported rotation")
        height, width = image.shape[:2]
        x, y, box_width, box_height = (float(item) for item in bbox)
        if not all(math.isfinite(item) for item in (x, y, box_width, box_height)):
            raise ValueError("invalid face box")
        if box_width < .06 or box_height < .06:
            raise ValueError("face too small")
        # Keep the face crop within the frame. The detector and this consumer
        # use the same logical source and rotation, so the normalized box aligns.
        pad_x, pad_y = .08 * box_width, .08 * box_height
        left, top = max(0, int((x - pad_x) * width)), max(0, int((y - pad_y) * height))
        right = min(width, int((x + box_width + pad_x) * width))
        bottom = min(height, int((y + box_height + pad_y) * height))
        if right - left < 32 or bottom - top < 32:
            raise ValueError("face crop too small")
        gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        tensor = resized.astype(np.float32).reshape(1, 1, 64, 64)
        self.net.setInput(tensor)
        return _softmax(self.net.forward().reshape(-1))


class CueSmoother:
    def __init__(self, *, window=5, required=3, threshold=.60, margin=.12):
        self.samples = collections.deque(maxlen=window)
        self.required = required
        self.threshold = threshold
        self.margin = margin
        self.track_id = None

    def reset(self):
        self.samples.clear()
        self.track_id = None

    def update(self, track_id, distribution):
        if track_id != self.track_id:
            self.samples.clear()
            self.track_id = track_id
        if set(distribution) != set(LABELS):
            raise ValueError("invalid cue distribution")
        self.samples.append(dict(distribution))
        if len(self.samples) < self.required:
            return None
        scores = {label: sum(item[label] for item in self.samples) / len(self.samples)
                  for label in LABELS}
        ordered = sorted(scores, key=scores.get, reverse=True)
        top, runner_up = ordered[:2]
        votes = sum(max(item, key=item.get) == top for item in self.samples)
        if votes < self.required or scores[top] < self.threshold or scores[top] - scores[runner_up] < self.margin:
            return None
        return {"track_id": track_id, "top_label": top,
                "confidence": round(scores[top], 4),
                "distribution": {key: round(value, 4) for key, value in scores.items()},
                "model": MODEL_ID}


class AffectWorker:
    """One front-camera consumer; never owns a device or saves frames."""

    def __init__(self, model, camera, face_status, context, *, clock=time.time,
                 monotonic=time.monotonic, fps=.5, rotation=180, resources=None):
        self.model = model
        self.camera = camera
        self.face_status = face_status
        self.context = context
        self.clock = clock
        self.monotonic = monotonic
        self.interval = 1 / fps
        self.rotation = rotation
        self.resources = resources
        self.smoother = CueSmoother()
        self.sequence = 0
        self.frames = 0
        self.inferences = 0
        self.resource_skips = 0
        self.last_success_at = None
        self.last_latency_ms = None
        self.last_error = None
        self.last_label = None

    def tick(self):
        started = self.monotonic()
        try:
            status = self.face_status()
            sources = [item for item in status.get("sources", [])
                       if item.get("source") == "camera.front" and item.get("ready")]
            if len(sources) != 1:
                self.smoother.reset(); self.last_label = None
                return
            source = sources[0]
            if self.clock() - float(source.get("last_frame_at") or 0) > 2:
                self.smoother.reset(); self.last_label = None
                return
            tracks = [item for item in source.get("tracks", [])
                      if item.get("state") == "confirmed" and
                      float(item.get("last_seen_seconds_ago", 99)) <= 1]
            # No cue when multiple faces could be confused, even if one dominates.
            if len(tracks) != 1:
                self.smoother.reset(); self.last_label = None
                return
            track = tracks[0]
            frame = self.camera.wait_frame(self.sequence, timeout=1)
            if frame is None:
                self.last_error = "frame_unavailable"
                return
            self.sequence, jpeg = frame
            self.frames += 1
            lease_id = None
            if self.resources is not None:
                try:
                    lease = self.resources.acquire(
                        ADMISSION_OWNER, ADMISSION_RESOURCES, "vision.affect",
                        priority=10, ttl_seconds=10,
                    )
                    lease_id = lease["lease_id"]
                except Exception:
                    # Affect is best-effort. A denied or unavailable lease
                    # drops this sample; it never delays an interactive turn.
                    self.resource_skips += 1
                    self.smoother.reset()
                    self.last_label = None
                    self.last_error = "resource_unavailable"
                    return
            try:
                distribution = self.model.infer(jpeg, track["bbox"], rotation=self.rotation)
            finally:
                if lease_id is not None:
                    try:
                        self.resources.release(lease_id, ADMISSION_OWNER)
                    except Exception:
                        # The short TTL bounds a lost release without blocking
                        # cue publication or extending the inference path.
                        pass
            self.inferences += 1
            cue = self.smoother.update(track["track_id"], distribution)
            now = self.clock()
            if cue is not None:
                self.context.observe(
                    provider="ferplus-cpu", kind="affect_cue", source="camera.front",
                    observed_at=now, ttl_seconds=3.0, confidence=cue["confidence"],
                    inference=True, data=cue,
                )
                self.last_label = cue["top_label"]
            else:
                self.last_label = None
            self.last_success_at = now
            self.last_latency_ms = round((self.monotonic() - started) * 1000, 2)
            self.last_error = None
        except Exception as exc:
            self.smoother.reset()
            self.last_label = None
            self.last_error = type(exc).__name__

    def status(self):
        return {"ready": self.last_success_at is not None and
                self.clock() - self.last_success_at < 5 and self.last_error is None,
                "model": MODEL_ID, "source": "camera.front", "sample_fps": round(1 / self.interval, 2),
                "frames": self.frames, "inferences": self.inferences,
                "resource_admission": "enabled" if self.resources is not None else "disabled",
                "resource_skips": self.resource_skips,
                "last_success_at": self.last_success_at, "last_latency_ms": self.last_latency_ms,
                "last_error": self.last_error, "cue": self.last_label,
                "privacy": {"raw_frames_retained": False, "affect_history_retained": False}}
