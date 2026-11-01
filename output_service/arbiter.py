"""Priority queue and lifecycle owner for all audible product output."""

from __future__ import annotations

import heapq
import re
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from concurrent.futures import Future

from gateway.speech.engine import prepare_tts_text
from gateway.speech.text import split_for_speech


OWNER = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
JOB_ID = re.compile(r"^output_[A-Za-z0-9_-]{20,64}$")
MEDIA_ID = re.compile(r"^(?:audio|generated)_[A-Za-z0-9_-]{20,64}$")
PLAYBACK_ID = re.compile(r"^playback_[A-Za-z0-9_-]{20,64}$")
CATEGORIES = {"conversation": 100, "alert": 60, "notification": 20}
TERMINAL = {"completed", "cancelled", "expired", "failed", "interrupted"}


class OutputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class OutputJob:
    job_id: str
    text: str
    media_id: str | None
    owner: str
    category: str
    priority: int
    volume_percent: int
    created_at: float
    speed: float = 1.0
    expires_at: float | None = None
    trace_id: str | None = None
    state: str = "queued"
    playback_id: str | None = None
    started_at: float | None = None
    ended_at: float | None = None
    error: str | None = None
    cancel_requested: bool = False
    guard: dict | None = None
    timing_ms: dict = field(default_factory=dict)
    streaming: bool = False
    segment_count: int = 1
    stream_stop_confirmed: bool = False

    def public(self):
        # Never export text, a guard, PCM or future/thread internals to events.
        fields = ("job_id", "media_id", "owner", "category", "priority", "volume_percent",
                  "created_at", "expires_at", "trace_id", "state", "playback_id", "speed",
                  "started_at", "ended_at", "error", "streaming", "segment_count")
        return {**{name: getattr(self, name) for name in fields}, "timing_ms": dict(self.timing_ms)}


class OutputArbiter:
    def __init__(self, engine, *, events=None, clock=time.time, monotonic=time.monotonic,
                 retention=256, segmented_tts=False):
        self.engine = engine
        self.events = events
        self.clock = clock
        self.monotonic = monotonic
        self.retention = retention
        self.condition = threading.Condition()
        self.queue = []
        self.jobs = OrderedDict()
        self.serial = 0
        self.current_id = None
        self.closed = False
        self.segmented_tts = bool(segmented_tts)
        self.quarantined = False
        self.degraded_reason = None
        self.worker = threading.Thread(target=self._run, name="vq-output-worker", daemon=True)

    def start(self):
        self.worker.start()

    def close(self):
        with self.condition:
            self.closed = True
            if self.current_id in self.jobs:
                self.jobs[self.current_id].cancel_requested = True
            self.condition.notify_all()
        # Await bounded speech transport and known media cleanup before exit.
        self.worker.join(timeout=45 if self.segmented_tts else 4)
        clean = not self.worker.is_alive() and not self.quarantined
        if not clean:
            self._quarantine("close_outcome_unconfirmed")
        return clean

    def say(self, *, text, owner, category="conversation", volume_percent=70,
            expires_in_seconds=None, trace_id=None, guard=None, speed=1.0):
        if type(speed) not in (int, float) or not 0.6 <= speed <= 1.0:
            raise OutputError("invalid_speed", "speed must be 0.6..1.0")
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise OutputError("invalid_text", "text must contain 1..2000 characters")
        if not isinstance(owner, str) or not OWNER.fullmatch(owner):
            raise OutputError("invalid_owner", "invalid owner")
        if category not in CATEGORIES:
            raise OutputError("invalid_category", "invalid output category")
        if not isinstance(volume_percent, int) or not 0 <= volume_percent <= 100:
            raise OutputError("invalid_volume", "volume must be 0..100")
        if expires_in_seconds is not None and (
                not isinstance(expires_in_seconds, (int, float)) or not 1 <= expires_in_seconds <= 600):
            raise OutputError("invalid_expiry", "expiry must be 1..600 seconds")
        now = self.clock()
        if guard is not None and (not isinstance(guard, dict) or set(guard) != {"kind", "token"}
                or guard.get("kind") not in {"posture", "greeting"} or category != "notification"
                or not isinstance(guard.get("token"), str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{20,64}", guard["token"])):
            raise OutputError("invalid_guard", "invalid bounded output guard")
        job = OutputJob(
            job_id="output_" + secrets.token_urlsafe(18), text=text.strip(), media_id=None, owner=owner,
            category=category, priority=CATEGORIES[category], volume_percent=volume_percent,
            created_at=now,
            speed=float(speed),
            expires_at=None if expires_in_seconds is None else now + expires_in_seconds,
            trace_id=trace_id,
            guard=dict(guard) if guard else None,
        )
        with self.condition:
            if self.closed or self.quarantined:
                raise OutputError("output_unavailable", "output is closed or quarantined")
            self.serial += 1
            self.jobs[job.job_id] = job
            heapq.heappush(self.queue, (-job.priority, self.serial, job.job_id))
            current = self.jobs.get(self.current_id)
            if current is not None and job.priority > current.priority:
                current.cancel_requested = True
            self._trim_locked()
            self.condition.notify_all()
        self._event("audio.output.queued", job)
        with self.condition:
            return {**job.public(), "accepted": True}

    def play_media(self, *, media_id, owner, category="conversation", volume_percent=70,
                   expires_in_seconds=None, trace_id=None):
        if not isinstance(media_id, str) or not MEDIA_ID.fullmatch(media_id):
            raise OutputError("invalid_media_id", "invalid media id")
        if not isinstance(owner, str) or not OWNER.fullmatch(owner):
            raise OutputError("invalid_owner", "invalid owner")
        if category not in CATEGORIES:
            raise OutputError("invalid_category", "invalid output category")
        if not isinstance(volume_percent, int) or not 0 <= volume_percent <= 100:
            raise OutputError("invalid_volume", "volume must be 0..100")
        if expires_in_seconds is not None and (
                not isinstance(expires_in_seconds, (int, float)) or not 1 <= expires_in_seconds <= 600):
            raise OutputError("invalid_expiry", "expiry must be 1..600 seconds")
        now = self.clock()
        job = OutputJob(
            job_id="output_" + secrets.token_urlsafe(18), text="", media_id=media_id,
            owner=owner, category=category, priority=CATEGORIES[category],
            volume_percent=volume_percent, created_at=now,
            expires_at=None if expires_in_seconds is None else now + expires_in_seconds,
            trace_id=trace_id,
        )
        with self.condition:
            if self.closed or self.quarantined:
                raise OutputError("output_unavailable", "output is closed or quarantined")
            self.serial += 1
            self.jobs[job.job_id] = job
            heapq.heappush(self.queue, (-job.priority, self.serial, job.job_id))
            current = self.jobs.get(self.current_id)
            if current is not None and job.priority > current.priority:
                current.cancel_requested = True
            self._trim_locked()
            self.condition.notify_all()
        self._event("audio.output.queued", job)
        with self.condition:
            return {**job.public(), "accepted": True}

    def status(self, *, job_id):
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            raise OutputError("invalid_job_id", "invalid job id")
        with self.condition:
            job = self.jobs.get(job_id)
            if job is None:
                raise OutputError("job_not_found", "output job not found")
            return job.public()

    def cancel(self, *, job_id):
        with self.condition:
            job = self.jobs.get(job_id)
            if job is None:
                raise OutputError("job_not_found", "output job not found")
            if job.state in TERMINAL:
                return job.public()
            job.cancel_requested = True
            if job.state == "queued":
                job.state = "cancelled"
                job.ended_at = self.clock()
            self.condition.notify_all()
            return job.public()

    def system_status(self):
        with self.condition:
            queued = sum(job.state == "queued" for job in self.jobs.values())
            current = self.jobs.get(self.current_id)
            return {"ready": not self.closed and not self.quarantined,
                    "output_quarantined": self.quarantined, "degraded_reason": self.degraded_reason,
                    "segmented_tts_enabled": self.segmented_tts,
                    "current": current.public() if current else None,
                    "queued": queued, "retained": len(self.jobs),
                    "policy": "priority-noninterrupting-low-output",
                    "priorities": dict(CATEGORIES)}

    def _trim_locked(self):
        while len(self.jobs) > self.retention:
            key, job = next(iter(self.jobs.items()))
            if job.state not in TERMINAL:
                break
            self.jobs.pop(key)

    def _event(self, topic, job):
        if self.events is None:
            return
        try:
            with self.condition:
                payload = job.public()
            self.events.publish(topic, "output", payload, privacy="local", trace_id=job.trace_id)
        except Exception:
            pass

    def _still_wanted_locked(self, job):
        if self.closed or job.cancel_requested:
            job.state = "cancelled"
            return False
        if job.expires_at is not None and self.clock() >= job.expires_at:
            job.state = "expired"
            return False
        return True

    def _preflight(self, job):
        with self.condition:
            if not self._still_wanted_locked(job):
                return False
        # The fusion host may query this arbiter to exclude this very job.
        # No remote call may hold condition, including this validator.
        if job.guard:
            try:
                allowed = self.engine.validate_guard(job.guard, job.job_id)
            except Exception:
                allowed = False
            if not allowed:
                with self.condition:
                    if self._still_wanted_locked(job):
                        job.state, job.error = "cancelled", "context_invalid"
                return False
        with self.condition:
            return self._still_wanted_locked(job)

    def _next(self):
        with self.condition:
            while not self.closed:
                while self.queue:
                    _priority, _serial, job_id = heapq.heappop(self.queue)
                    job = self.jobs[job_id]
                    if job.state != "queued":
                        continue
                    if self.quarantined:
                        job.state, job.error, job.ended_at = "failed", self.degraded_reason, self.clock()
                        self._event("audio.output.failed", job)
                        continue
                    if job.expires_at is not None and self.clock() >= job.expires_at:
                        job.state, job.ended_at = "expired", self.clock()
                        self._event("audio.output.expired", job)
                        continue
                    self.current_id = job_id
                    job.state, job.started_at = "synthesizing", self.clock()
                    job.timing_ms["queue"] = round((job.started_at - job.created_at) * 1000, 2)
                    return job
                self.condition.wait()
            return None

    def _run(self):
        while True:
            job = self._next()
            if job is None:
                return
            media_id = job.media_id
            generated = media_id is None
            lease = None
            try:
                if generated and self._try_segmented(job):
                    continue
                if not self._preflight(job):
                    continue
                self._event("audio.output.synthesizing", job)
                synthesis_started = self.monotonic()
                if generated:
                    media_id = self.engine.synthesize(job.text, job.owner,
                        **({"speed": job.speed} if job.speed != 1 else {}))["media_id"]
                with self.condition:
                    job.timing_ms["synthesis"] = round((self.monotonic() - synthesis_started) * 1000, 2)
                if not self._preflight(job):
                    continue
                if job.guard:
                    lease = self.engine.reserve_output(job.owner)
                # Both validation and lease acquisition may take time. Recheck
                # after acquiring the lease, immediately before the effect.
                if not self._preflight(job):
                    continue
                with self.condition:
                    if not self._still_wanted_locked(job):
                        continue
                # Keep current_id/state active, but let status/cancel/high-
                # priority enqueue progress during a slow Runtime submission.
                playback_started = self.monotonic()
                playback = self.engine.start_playback(
                    media_id, job.owner, volume_percent=job.volume_percent)
                with self.condition:
                    job.timing_ms["playback_submit"] = round((self.monotonic() - playback_started) * 1000, 2)
                    job.timing_ms["to_playback"] = round((self.clock() - job.created_at) * 1000, 2)
                    job.timing_ms["playback_lead_in"] = playback.get("lead_in_ms", 0)
                    job.playback_id = playback["playback_id"]
                    if self._still_wanted_locked(job):
                        job.state = "playing"
                    else:
                        if job.state == "cancelled" and self._has_higher(job.priority):
                            job.state = "interrupted"
                if job.state != "playing":
                    self.engine.stop_playback(job.playback_id)
                    continue
                if job.state == "playing":
                    self._event("audio.output.started", job)
                while job.state == "playing":
                    if job.cancel_requested:
                        self.engine.stop_playback(job.playback_id)
                        job.state = "interrupted" if self._has_higher(job.priority) else "cancelled"
                        break
                    state = self.engine.playback_status(job.playback_id).get("state", "failed")
                    if state != "playing":
                        job.state = "completed" if state == "completed" else "failed"
                        break
                    time.sleep(0.1)
            except Exception as exc:
                job.state = "failed"
                job.error = type(exc).__name__
            finally:
                if lease is not None:
                    try:
                        self.engine.release_output(lease, job.owner)
                    except Exception:
                        pass
                if generated and media_id is not None:
                    try:
                        self.engine.release(media_id, job.owner)
                    except Exception:
                        pass
                with self.condition:
                    job.ended_at = self.clock()
                    self.current_id = None
                    self._trim_locked()
                    self.condition.notify_all()
                self._event("audio.output." + job.state, job)

    def _quarantine(self, reason):
        # A transport exception is not a native completion/cancellation receipt.
        # Reject new audible jobs until an operator verifies Runtime and restarts
        # this optional output path; never spawn more generators to hide failure.
        with self.condition:
            self.quarantined, self.degraded_reason = True, reason
            self.condition.notify_all()

    @staticmethod
    def _unknown(exc):
        return getattr(exc, "code", None) in {"TRANSPORT_ERROR", "PROTOCOL_ERROR"} or isinstance(exc, (OSError, TimeoutError))

    def _wanted(self, job):
        with self.condition:
            return not (self.closed or job.cancel_requested or
                        (job.expires_at is not None and self.clock() >= job.expires_at))

    def _stop_stream(self, job):
        if job.playback_id is None or job.stream_stop_confirmed:
            return
        try:
            result = self.engine.stop_playback(job.playback_id)
            if not isinstance(result, dict) or result.get("state") not in {"stopped", "completed", "failed"}:
                raise OutputError("stop_unconfirmed", "stream stop did not confirm termination")
            job.stream_stop_confirmed = True
        except Exception:
            self._quarantine("playback_stop_unconfirmed")
            raise

    def _segment_synthesis(self, job, text):
        future = Future()
        def generate():
            try:
                future.set_result(self.engine.synthesize(text, job.owner,
                    **({"speed": job.speed} if job.speed != 1 else {})))
            except Exception as exc:
                future.set_exception(exc)
            finally:
                with self.condition:
                    self.condition.notify_all()
        # Only one pending synthesis per job; no queue or resident model/daemon.
        thread = threading.Thread(target=generate, name="vq-output-segment", daemon=True)
        thread.start()
        stopped = False
        while not future.done():
            if not self._wanted(job) and job.playback_id and not stopped:
                try:
                    self._stop_stream(job)
                except Exception:
                    # Still wait for the known RPC task; do not launch another.
                    pass
                stopped = True
            with self.condition:
                self.condition.wait(.05)
        thread.join()
        try:
            result = future.result()
            if not isinstance(result, dict) or not isinstance(result.get("media_id"), str) or not MEDIA_ID.fullmatch(result["media_id"]):
                raise OutputError("PROTOCOL_ERROR", "synthesis acknowledgement has no valid media ID")
        except Exception as exc:
            if self._unknown(exc):
                self._quarantine("native_synthesis_outcome_unknown")
            raise
        if not self._wanted(job):
            self.engine.release(result["media_id"], job.owner)
            self._preflight(job)
            return None
        return result

    @staticmethod
    def _start_ack(value):
        if not isinstance(value, dict) or not isinstance(value.get("playback_id"), str) or not PLAYBACK_ID.fullmatch(value["playback_id"]) or value.get("state") not in {"playing", "stopped", "failed", "completed"}:
            raise OutputError("PROTOCOL_ERROR", "stream start acknowledgement is invalid")
        return value

    def _stream_state(self, job):
        state = self.engine.playback_status(job.playback_id)
        if state.get("state") != "playing":
            if state.get("state") in {"stopped", "failed", "completed"}:
                job.stream_stop_confirmed = True
            raise OutputError("stream_ended_early", "stream ended before input was finished")
        return state

    def _try_segmented(self, job):
        if not (self.segmented_tts and job.category == "conversation" and job.guard is None):
            return False
        text = prepare_tts_text(job.text)
        segments = split_for_speech(text)
        if len(segments) < 2 or not self.engine.streaming_available():
            return False  # Before any effect: legacy single-WAV path only.
        with self.condition:
            job.streaming, job.segment_count = True, len(segments)
        if not self._preflight(job):
            return True
        self._event("audio.output.synthesizing", job)
        started, synthesis_ms = self.monotonic(), 0.0
        stream_finished = False
        try:
            for index, segment in enumerate(segments):
                if not self._preflight(job):
                    return True
                # pending is a bounded software FD/feed window, not proof of
                # acoustic completion. Include one pending native call in it.
                while job.playback_id and self._stream_state(job).get("pending_segments", 0) >= 2:
                    if not self._preflight(job):
                        return True
                    time.sleep(.1)
                synth_started = self.monotonic()
                result = self._segment_synthesis(job, segment)
                synthesis_ms += (self.monotonic() - synth_started) * 1000
                if result is None:
                    return True
                media_id = result["media_id"]
                try:
                    if not self._preflight(job):
                        return True
                    if index == 0:
                        submitting = self.monotonic()
                        try:
                            playback = self._start_ack(self.engine.start_stream_playback(
                                media_id, job.owner, stream_key=job.job_id, volume_percent=job.volume_percent))
                        except Exception as exc:
                            if not self._unknown(exc):
                                raise
                            # Exact owner+key+media idempotence, once only.
                            try:
                                playback = self._start_ack(self.engine.start_stream_playback(
                                    media_id, job.owner, stream_key=job.job_id,
                                    volume_percent=job.volume_percent, resolve_only=True))
                            except Exception:
                                self._quarantine("stream_start_outcome_unknown")
                                raise
                        with self.condition:
                            job.playback_id = playback["playback_id"]
                            job.timing_ms.update(
                                synthesis=round(synthesis_ms, 2),
                                first_segment_synthesis=round(synthesis_ms, 2),
                                playback_submit=round((self.monotonic() - submitting) * 1000, 2),
                                to_playback=round((self.clock() - job.created_at) * 1000, 2),
                                playback_lead_in=playback.get("lead_in_ms", 0))
                        if playback.get("state") != "playing":
                            if playback.get("state") in {"stopped", "failed", "completed"}:
                                job.stream_stop_confirmed = True
                            raise OutputError("stream_start_failed", "stream did not start")
                        if not self._preflight(job):
                            return True
                        with self.condition:
                            job.state = "playing"
                        self._event("audio.output.started", job)
                    else:
                        self._stream_state(job)
                        for attempt in range(6):
                            if not self._preflight(job):
                                return True
                            try:
                                ack = self.engine.append_stream_playback(job.playback_id, media_id, job.owner, index)
                                if not isinstance(ack, dict) or ack.get("state") != "playing" or ack.get("accepted_segments", 0) < index + 1:
                                    raise OutputError("PROTOCOL_ERROR", "segment acceptance was not acknowledged")
                                break
                            except Exception as exc:
                                # Only a definite not-accepted capacity response
                                # may retry; a lost response is never replayed.
                                details = getattr(exc, "details", {}) or {}
                                if not (getattr(exc, "code", None) == "TIMEOUT" and
                                        details.get("reason") == "stream_capacity" and attempt < 5):
                                    raise
                                time.sleep(.1)
                finally:
                    # ACK owns an opened WAV FD in Runtime. Even unknown ACK is
                    # safe to unlink: accepted reads retain that FD; rejected
                    # reads have not taken the media. No caller-owned playback.
                    self.engine.release(media_id, job.owner)
            if not self._preflight(job):
                return True
            finish = self.engine.finish_stream_playback(job.playback_id, job.owner)
            if finish.get("state") in {"stopped", "failed", "completed"}:
                job.stream_stop_confirmed = True
                if finish.get("state") != "completed" or not finish.get("input_finished"):
                    raise OutputError("stream_ended_early", "stream ended before finish acknowledgement")
                with self.condition:
                    job.state = "completed"
                    job.timing_ms.update(synthesis=round(synthesis_ms, 2),
                        producer_wait=finish.get("producer_wait_ms", 0),
                        stream_total=round((self.monotonic() - started) * 1000, 2))
                stream_finished = True
                return True
            if not finish.get("input_finished"):
                raise OutputError("finish_unconfirmed", "stream input finish was not acknowledged")
            while self._preflight(job):
                state = self.engine.playback_status(job.playback_id)
                if state.get("state") != "playing":
                    if state.get("state") in {"stopped", "failed", "completed"}:
                        job.stream_stop_confirmed = True
                    if state.get("state") != "completed" or not state.get("input_finished"):
                        raise OutputError("stream_incomplete", "stream failed before drain completion")
                    with self.condition:
                        job.state = "completed"
                        job.timing_ms.update(synthesis=round(synthesis_ms, 2),
                            producer_wait=state.get("producer_wait_ms", 0),
                            stream_total=round((self.monotonic() - started) * 1000, 2))
                    stream_finished = True
                    break
                time.sleep(.1)
            return True
        finally:
            if job.playback_id and not stream_finished:
                self._stop_stream(job)

    def _has_higher(self, priority):
        with self.condition:
            return any(job.state == "queued" and job.priority > priority for job in self.jobs.values())
