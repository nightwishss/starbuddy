"""CPU-first SFace enrollment and recognition pipeline."""

from __future__ import annotations

import re
import secrets
import threading
import time

from camera_stream_service import CameraStreamClient
from event_service import EventClient
from face_vault_service.client import FaceVaultClient
from output_service import OutputClient


MODEL_ID = "opencv-sface-2021dec-fp32"
CONSENT_VERSION = "starbuddy-face-enrollment-v1"
PROFILE_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")


class IdentityError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message); self.code = code


class SFaceEngine:
    def __init__(self, detector_path, recognizer_path, *, rotation=180):
        import cv2
        cv2.setNumThreads(1)
        self.cv2, self.rotation = cv2, rotation
        self.detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), .82, .3, 100)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")

    def analyze(self, jpeg):
        import numpy as np
        cv2 = self.cv2
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise IdentityError("invalid_frame", "camera frame is not a JPEG")
        if self.rotation == 180: image = cv2.rotate(image, cv2.ROTATE_180)
        elif self.rotation == 90: image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 270: image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _unused, rows = self.detector.detect(image)
        candidates = []
        for row in ([] if rows is None else rows):
            x, y, box_width, box_height = [float(value) for value in row[:4]]
            score = float(row[14]); area_ratio = max(0.0, box_width * box_height / (width * height))
            aligned = self.recognizer.alignCrop(image, row)
            sharpness = float(cv2.Laplacian(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            eye_midpoint = (float(row[4]) + float(row[6])) / 2
            eye_distance = max(1.0, abs(float(row[6]) - float(row[4])))
            yaw_proxy = (float(row[8]) - eye_midpoint) / eye_distance
            pose = "left" if yaw_proxy < -.10 else "right" if yaw_proxy > .10 else "center"
            reasons = []
            if score < .85: reasons.append("low_detection_confidence")
            if area_ratio < .025: reasons.append("face_too_small")
            if sharpness < 24: reasons.append("image_too_blurry")
            feature = self.recognizer.feature(aligned).flatten().astype("float32").tolist()
            quality = min(1.0, .5 * score + .3 * min(1.0, area_ratio / .10) + .2 * min(1.0, sharpness / 120))
            candidates.append({
                "bbox": [round(max(0.0, x / width), 5), round(max(0.0, y / height), 5),
                         round(min(1.0, box_width / width), 5), round(min(1.0, box_height / height), 5)],
                "score": round(score, 4), "quality": round(quality, 4), "sharpness": round(sharpness, 1),
                "pose": pose, "acceptable": not reasons, "rejection_reasons": reasons, "embedding": feature,
            })
        candidates.sort(key=lambda value: value["bbox"][2] * value["bbox"][3], reverse=True)
        return candidates


class IdentityService:
    POSES = ("center", "left", "right")

    def __init__(self, engine, *, camera=None, vault=None, events=None, fps=2.0,
                 output=None, greeting_cooldown=120.0,
                 monotonic=time.monotonic, wall_clock=time.time):
        self.engine = engine
        self.camera = camera or CameraStreamClient(consumer="face-identity", source="camera.front", rotation=0)
        self.vault = vault or FaceVaultClient()
        self.events = events or EventClient(timeout=1)
        self.output = output or OutputClient(timeout=3)
        self.greeting_cooldown = float(greeting_cooldown)
        self.fps, self.monotonic, self.wall_clock = fps, monotonic, wall_clock
        self.lock = threading.RLock(); self.closed = threading.Event(); self.sequence = 0
        self.thread = threading.Thread(target=self._run, name="vq-face-identity", daemon=True)
        self.enrollment = None; self.observation = {"status": "starting"}; self.error = None
        self.last_frame_at = None; self.latency_ms = None; self.candidate_key = None; self.candidate_hits = 0
        self.stable_identity = None; self.last_published_identity = None
        self.last_spoken = {}; self.last_speech = None

    def start(self): self.thread.start()
    def close(self): self.closed.set(); self.thread.join(timeout=5)

    @staticmethod
    def _clean_label(value):
        if not isinstance(value, str) or not value.strip() or len(value.strip()) > 96:
            raise IdentityError("invalid_label", "name must contain 1 to 96 characters")
        return value.strip()

    def start_enrollment(self, *, label, profile_id=None, consent=None, consent_version=None):
        if consent not in {"user_confirmed", "guardian_confirmed"} or consent_version != CONSENT_VERSION:
            raise IdentityError("consent_required", "the current local enrollment consent must be confirmed")
        label = self._clean_label(label)
        if profile_id is not None and (not isinstance(profile_id, str) or not PROFILE_RE.fullmatch(profile_id)):
            raise IdentityError("invalid_profile_id", "invalid profile_id")
        with self.lock:
            if self.enrollment is not None:
                raise IdentityError("enrollment_active", "an enrollment session is already active")
            self.enrollment = {"token": secrets.token_urlsafe(24), "label": label, "profile_id": profile_id,
                               "consent": consent, "consent_version": consent_version, "samples": [],
                               "started_at": self.wall_clock(), "expires_at": self.monotonic() + 120,
                               "last_sample_at": 0.0, "state": "collecting", "message": "请正视摄像头"}
            result = self._enrollment_public(include_token=True)
        self._say("好的，请正视摄像头", "enrollment:start:" + result["token"], cooldown=0)
        return result

    def cancel_enrollment(self, token):
        with self.lock:
            session = self._require_session(token)
            if session["state"] == "committing":
                raise IdentityError("commit_in_progress", "enrollment commit is already in progress")
            self.enrollment = None
        self._say("已取消注册", "enrollment:cancel", cooldown=2)
        return {"cancelled": True}

    def commit_enrollment(self, token, *, consent):
        if consent not in {"user_confirmed", "guardian_confirmed"}:
            raise IdentityError("consent_required", "saving biometric templates requires confirmation")
        with self.lock:
            session = self._require_session(token)
            if session["state"] != "ready" or len(session["samples"]) != len(self.POSES):
                raise IdentityError("enrollment_not_ready", "three accepted poses are required")
            samples = list(session["samples"]); session["state"] = "committing"
        identity = None
        try:
            identity = self.vault.create_identity(consent=consent, consent_version=CONSENT_VERSION,
                                                  profile_id=session["profile_id"], label=session["label"])
            for sample in samples:
                self.vault.add_template(identity["identity_id"], MODEL_ID, sample["embedding"],
                                        consent=consent, quality=sample["quality"])
        except Exception as exc:
            if identity is not None:
                try: self.vault.delete_identity(identity["identity_id"], consent=consent)
                except Exception: pass
            with self.lock:
                if self.enrollment is session: session["state"] = "ready"
            raise IdentityError("vault_write_failed", type(exc).__name__) from exc
        with self.lock:
            if self.enrollment is session: self.enrollment = None
        self._say(f"{session['label']}，你好，我已经记住你了", "enrollment:committed", cooldown=2)
        return {"enrolled": True, "identity": self.vault.identity(identity["identity_id"]),
                "raw_frames_retained": False}

    def delete_identity(self, identity_id, *, consent, confirmation):
        if confirmation != "DELETE_IDENTITY":
            raise IdentityError("confirmation_required", "confirmation must be DELETE_IDENTITY")
        if consent not in {"user_confirmed", "guardian_confirmed"}:
            raise IdentityError("consent_required", "identity deletion requires confirmation")
        try:
            result = self.vault.delete_identity(identity_id, consent=consent)
        except Exception as exc:
            raise IdentityError("vault_delete_failed", type(exc).__name__) from exc
        self._say("这个身份已经删除", "identity:deleted", cooldown=2)
        return result

    def _require_session(self, token):
        if self.enrollment is None or not isinstance(token, str) or not secrets.compare_digest(token, self.enrollment["token"]):
            raise IdentityError("invalid_enrollment", "invalid or expired enrollment session")
        return self.enrollment

    def _enrollment_public(self, *, include_token=False):
        if self.enrollment is None: return None
        value = self.enrollment
        result = {"label": value["label"], "state": value["state"],
                  "sample_count": len(value["samples"]), "required_samples": len(self.POSES),
                  "accepted_poses": [item["pose"] for item in value["samples"]],
                  "next_pose": self.POSES[min(len(value["samples"]), len(self.POSES) - 1)],
                  "message": value["message"], "started_at": value["started_at"]}
        if include_token: result["token"] = value["token"]
        return result

    def status(self):
        with self.lock:
            age = None if self.last_frame_at is None else max(0.0, self.wall_clock() - self.last_frame_at)
            return {"ready": self.error is None and age is not None and age < 3, "model": MODEL_ID,
                    "source": "camera.front", "sequence": self.sequence, "last_frame_seconds_ago": age,
                    "latency_ms": self.latency_ms, "observation": dict(self.observation),
                    "enrollment": self._enrollment_public(), "error": self.error,
                    "last_speech": None if self.last_speech is None else dict(self.last_speech),
                    "recognition_response": "context_only",
                    "privacy": {"raw_frames_retained": False, "embeddings_returned": False,
                                "cloud_used": False, "authentication_grade": False}}

    def identities(self): return self.vault.identities()

    def debug_frame(self):
        value = self.camera.wait_frame(0, timeout=3)
        if value is None: raise IdentityError("frame_unavailable", "camera frame unavailable")
        return value

    def _run(self):
        interval = 1 / self.fps
        while not self.closed.is_set():
            started = self.monotonic()
            try:
                value = self.camera.wait_frame(self.sequence, timeout=3)
                if value is None: raise IdentityError("frame_unavailable", "camera frame unavailable")
                sequence, jpeg = value; candidates = self.engine.analyze(jpeg)
                self._process(sequence, candidates, started)
            except Exception as exc:
                with self.lock: self.error = type(exc).__name__
            self.closed.wait(max(0.0, interval - (self.monotonic() - started)))

    def _process(self, sequence, candidates, started):
        now = self.monotonic()
        with self.lock:
            self.sequence = sequence; self.last_frame_at = self.wall_clock()
            self.latency_ms = round((now - started) * 1000, 2); self.error = None
            session = self.enrollment
            if session is not None:
                if now >= session["expires_at"]:
                    self.enrollment = None; self.observation = {"status": "enrollment_expired"}; return
                self._collect(session, candidates, now)
                self.observation = self._public_candidate(candidates[0]) if candidates else {"status": "no_face"}
                return
        self._recognize(candidates)

    def _collect(self, session, candidates, now):
        if session["state"] != "collecting": return
        if len(candidates) != 1:
            session["message"] = "请确保画面中只有一张脸"; return
        candidate = candidates[0]
        if not candidate["acceptable"]:
            session["message"] = "画面质量不足：" + ", ".join(candidate["rejection_reasons"]); return
        expected = self.POSES[len(session["samples"])]
        if candidate["pose"] != expected:
            names = {"center": "正视", "left": "稍向左转头", "right": "稍向右转头"}
            session["message"] = "请" + names[expected]; return
        if now - session["last_sample_at"] < .7: return
        session["samples"].append({"pose": expected, "quality": candidate["quality"],
                                   "embedding": list(candidate["embedding"])})
        session["last_sample_at"] = now
        if len(session["samples"]) == len(self.POSES):
            session["state"] = "ready"; session["message"] = "采集完成，请确认保存"
            self._say("采集完成，请点击确认保存", "enrollment:ready", cooldown=2)
        else:
            names = {"center": "正视", "left": "稍向左转头", "right": "稍向右转头"}
            session["message"] = "很好，接下来请" + names[self.POSES[len(session["samples"])]]
            speech = "很好，请稍向左转头" if len(session["samples"]) == 1 else "很好，请稍向右转头"
            self._say(speech, f"enrollment:pose:{len(session['samples'])}", cooldown=2)

    @staticmethod
    def _public_candidate(candidate):
        return {key: candidate[key] for key in ("bbox", "score", "quality", "sharpness", "pose", "acceptable", "rejection_reasons")}

    def _recognize(self, candidates):
        if not candidates:
            match = {"status": "no_face"}
        elif not candidates[0]["acceptable"]:
            match = {"status": "low_quality", **self._public_candidate(candidates[0])}
        else:
            candidate = candidates[0]
            match = self.vault.match(MODEL_ID, candidate["embedding"])
            match.update({"bbox": candidate["bbox"], "quality": candidate["quality"]})
        key = match.get("identity_id") if match.get("status") == "known" else match.get("status")
        with self.lock:
            if key == self.candidate_key: self.candidate_hits += 1
            else: self.candidate_key, self.candidate_hits = key, 1
            if self.candidate_hits >= 3:
                self.observation = {k: v for k, v in match.items() if k != "embedding"}
                if match.get("status") == "known" and match.get("identity_id") != self.last_published_identity:
                    self.last_published_identity = match["identity_id"]
                    self._publish(match)
                    # Recognition/status/events remain available to Fusion;
                    # recognition alone must never enqueue a spoken greeting.
                elif match.get("status") != "known":
                    self.last_published_identity = None
            else:
                self.observation = {"status": "confirming", "hits": self.candidate_hits,
                                    "required_hits": 3, "bbox": match.get("bbox")}

    def _publish(self, match):
        try:
            self.events.publish("perception.identity.recognized", "face-identity-service",
                                {key: match.get(key) for key in ("identity_id", "profile_id", "label", "similarity")},
                                privacy="local-private")
        except Exception:
            pass

    def _say(self, text, key, *, cooldown):
        if key.startswith("recognition:"):
            return None
        now = self.monotonic()
        with self.lock:
            previous = self.last_spoken.get(key)
            if previous is not None and now - previous < cooldown:
                return None
            self.last_spoken[key] = now
        try:
            result = self.output.say(text, "session:face-identity", category="notification",
                                     volume_percent=85, expires_in_seconds=15)
            summary = {"key": key, "job_id": result.get("job_id"), "queued_at": self.wall_clock(),
                       "error": None}
        except Exception as exc:
            summary = {"key": key, "job_id": None, "queued_at": self.wall_clock(),
                       "error": type(exc).__name__}
        with self.lock:
            self.last_speech = summary
        return summary
