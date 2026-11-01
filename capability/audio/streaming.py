"""Bounded, interruptible WAV segments feeding one PipeWire PCM process.

Media stays on its owned file descriptor; only small PCM blocks reach stdin.
The public ``playing`` state is software liveness, never an acoustic onset.
"""

from __future__ import annotations

import os
import re
import secrets
import select
import signal
import subprocess
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from typing import BinaryIO

from runtime.errors import (
    CapabilityError, HardwareError, PermissionDeniedError, ResourceNotFoundError, SchemaValidationError,
    TimeoutError as CapabilityTimeoutError,
)

from . import physical

_IDLE_SECONDS = 35.0
_MAX_WALL_SECONDS = 150.0
_MAX_SEGMENTS = 16
_MAX_PENDING = 2
_CAPACITY_WAIT_SECONDS = 0.25
_WRITE_SECONDS = 3.0
_DRAIN_SECONDS = 10.0
_POLL_SECONDS = 0.05
_BLOCK_BYTES = 4096
_STREAM_KEY = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_RECEIPT_SECONDS = 600.0
_MAX_RECEIPTS = 64
_RECEIPTS = {}
_START_LOCK = threading.Lock()


@dataclass
class Segment:
    media_id: str
    raw: BinaryIO
    reader: wave.Wave_read
    sample_rate: int
    channels: int
    frames: int

    def close(self):
        try:
            self.reader.close()
        finally:
            self.raw.close()


def _open_segment(media_id: str, owner: str, capability: str) -> Segment:
    """Validate the header/tail, then retain the open WAV through ACK/unlink."""
    path, info = physical._shared_wav(media_id, owner, capability)
    raw = reader = None
    retained = False
    try:
        raw = path.open("rb")
        reader = wave.open(raw, "rb")
        rate, channels, width, frames = (
            reader.getframerate(), reader.getnchannels(),
            reader.getsampwidth(), reader.getnframes(),
        )
        if (reader.getcomptype() != "NONE" or width != 2
                or not 8000 <= rate <= 192000 or channels not in {1, 2} or frames <= 0):
            raise wave.Error("unsupported s16 PCM WAV")
        if frames / rate > physical._MAX_PLAYBACK_SECONDS:
            raise SchemaValidationError("播放音频超过120秒限制", capability=capability)
        reader.setpos(frames - 1)
        if len(reader.readframes(1)) != channels * width:
            raise wave.Error("truncated PCM WAV")
        if reader.readframes(1):
            raise wave.Error("unaligned PCM data")
        reader.rewind()
        if info["media_id"] != media_id:
            raise HardwareError("Media Service媒体标识不一致", capability=capability)
        segment = Segment(media_id, raw, reader, rate, channels, frames)
        retained = True
        return segment
    except (OSError, EOFError, KeyError, wave.Error) as exc:
        raise HardwareError("连续播放WAV无效", capability=capability) from exc
    finally:
        if raw is not None and not retained:
            if reader is not None:
                reader.close()
            raw.close()


def _terminate(process):
    """Bound cancellation latency; stdin and media are finalized by feeder."""
    try:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.25)
        else:
            process.wait(timeout=0.25)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=0.25)
        except (OSError, subprocess.TimeoutExpired):
            pass
    return process.poll() is not None


class StreamPlayback(physical.Playback):
    streaming = True

    def __init__(self, playback_id, owner, first, role, device_id, process, lead_in_ms, stream_key):
        super().__init__(playback_id, owner, first.media_id, role, device_id,
                         first.frames / first.sample_rate + lead_in_ms / 1000,
                         time.time(), process, lead_in_ms=lead_in_ms)
        self.sample_rate = first.sample_rate
        self.stream_key = stream_key
        self.channels = first.channels
        self._frames = first.frames
        self._accepted = [first.media_id]
        self._pending = deque([first])
        self._active = None
        self._condition = threading.Condition(threading.RLock())
        self._producer_lock = threading.Lock()
        self._cancel = threading.Event()
        self._finalized = threading.Event()
        self._state = "playing"
        self._failure = None
        self._input_finished = False
        self._producer_wait_ms = 0.0
        self._deadline = time.monotonic() + _MAX_WALL_SECONDS + lead_in_ms / 1000
        self._thread = threading.Thread(target=self._feed, name="vq-audio-pcm", daemon=True)

    def start(self):
        self._thread.start()

    def _pending_count(self):
        return len(self._pending) + int(self._active is not None)

    def _check_owner(self, owner, capability):
        physical._validate_owner(owner, capability)
        if not secrets.compare_digest(owner, self.owner):
            raise PermissionDeniedError("无权写入该连续播放", capability=capability)

    def _check_live(self, capability):
        if self._state != "playing":
            raise HardwareError("连续播放已经结束", capability=capability,
                                details={"state": self._state, "reason": self._failure})

    def public(self):
        with self._condition:
            returncode = self.process.poll()
            state = self._state
            # A terminal status must mean both child exit and descriptor cleanup.
            if state != "playing" and (returncode is None or not self._finalized.is_set()):
                state = "playing"
            result = {
                "playback_id": self.playback_id, "media_id": self.media_id,
                "stream_key": self.stream_key,
                "role": self.role, "device_id": self.device_id, "state": state,
                "duration_seconds": round(self.duration_seconds, 3),
                "elapsed_seconds": round(max(0.0, time.time() - self.started_at), 3),
                "lead_in_ms": self.lead_in_ms,
                "accepted_segments": len(self._accepted),
                "pending_segments": self._pending_count(),
                "producer_wait_ms": round(self._producer_wait_ms, 3),
                "input_finished": self._input_finished,
            }
            if returncode is not None:
                result["returncode"] = returncode
            if self._failure:
                result["failure_reason"] = self._failure
            return result

    def add(self, media_id, owner, index):
        capability = "audio.append_stream_playback"
        self._check_owner(owner, capability)
        with self._producer_lock:
            try:
                with self._condition:
                    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
                        raise SchemaValidationError("segment_index必须是从1开始的整数", capability=capability)
                    if index < len(self._accepted) and self._accepted[index] == media_id:
                        return self.public()
                    self._check_live(capability)
                    if self._input_finished or index != len(self._accepted):
                        raise SchemaValidationError("连续播放段序号冲突或输入已结束", capability=capability)
                    if len(self._accepted) >= _MAX_SEGMENTS:
                        raise SchemaValidationError("连续播放最多16段", capability=capability)
                    waited = time.monotonic()
                    deadline = waited + _CAPACITY_WAIT_SECONDS
                    while self._pending_count() >= _MAX_PENDING:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            self._producer_wait_ms += (time.monotonic() - waited) * 1000
                            raise CapabilityTimeoutError(
                                "连续播放输入窗口已满", capability=capability,
                                details={"reason": "stream_capacity", "retryable": True,
                                         "pending_segments": self._pending_count()})
                        self._condition.wait(min(remaining, _POLL_SECONDS))
                        self._check_live(capability)
                    self._producer_wait_ms += (time.monotonic() - waited) * 1000
                # No capacity waiter holds a Media FD. Only this producer may
                # reserve a slot; stop and the feeder remain free during resolve.
                segment = _open_segment(media_id, owner, capability)
                try:
                    with self._condition:
                        self._check_live(capability)
                        if self._input_finished:
                            raise SchemaValidationError("连续播放输入已结束", capability=capability)
                        if (segment.sample_rate, segment.channels) != (self.sample_rate, self.channels):
                            raise SchemaValidationError("连续播放WAV格式必须一致", capability=capability)
                        if self._frames + segment.frames > self.sample_rate * physical._MAX_PLAYBACK_SECONDS:
                            raise SchemaValidationError("连续播放音频总量超过120秒", capability=capability)
                        self._frames += segment.frames
                        self.duration_seconds = self._frames / self.sample_rate + self.lead_in_ms / 1000
                        self._accepted.append(segment.media_id)
                        self._pending.append(segment)
                        self._condition.notify_all()
                        return self.public()
                except BaseException:
                    segment.close()
                    raise
            except CapabilityTimeoutError:
                # Capacity is retryable and leaves the accepted stream intact.
                raise
            except CapabilityError as exc:
                self.fail(exc.code.lower())
                raise
            except Exception as exc:
                self.fail("producer_failed")
                raise HardwareError("连续播放输入处理失败", capability=capability) from exc

    def finish(self, owner):
        self._check_owner(owner, "audio.finish_stream_playback")
        with self._condition:
            if self._input_finished:
                return self.public()
            self._check_live("audio.finish_stream_playback")
            self._input_finished = True
            self._condition.notify_all()
            return self.public()

    def _end(self, state, reason=None):
        with self._condition:
            if self._state == "playing":
                self._state, self._failure = state, reason
            self._cancel.set()
            self._condition.notify_all()
        _terminate(self.process)
        if threading.current_thread() is not self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.process.poll() is None or not self._finalized.is_set():
            with self._condition:
                self._failure = "unknown_stop"
            raise HardwareError("连续播放终止尚未确认", capability="audio.stop_playback",
                                details={"reason": "unknown_stop", "playback_id": self.playback_id})

    def fail(self, reason):
        self._end("failed", reason)

    def stop(self):
        self.stopped = True
        self._end("stopped")
        return self.public()

    def _write(self, data):
        remaining = memoryview(data)
        deadline = min(self._deadline, time.monotonic() + _WRITE_SECONDS)
        fd = self.process.stdin.fileno()
        while remaining:
            if self._cancel.is_set():
                raise InterruptedError("playback cancelled")
            if self.process.poll() is not None:
                raise BrokenPipeError("PipeWire exited before input drained")
            if time.monotonic() >= deadline:
                raise TimeoutError("PCM write deadline")
            try:
                count = os.write(fd, remaining)
                if count <= 0:
                    raise BrokenPipeError("empty PCM pipe write")
                remaining = remaining[count:]
            except BlockingIOError:
                select.select([], [fd], [], min(_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
            except InterruptedError:
                if self._cancel.is_set():
                    raise

    def _feed(self):
        completed = False
        try:
            silence_bytes = (self.sample_rate * self.lead_in_ms // 1000) * self.channels * 2
            while silence_bytes:
                count = min(_BLOCK_BYTES, silence_bytes)
                self._write(bytes(count))
                silence_bytes -= count
            while not self._cancel.is_set():
                with self._condition:
                    idle_deadline = time.monotonic() + _IDLE_SECONDS
                    while not self._pending and not self._input_finished:
                        if self._cancel.is_set():
                            raise InterruptedError("playback cancelled")
                        if self.process.poll() is not None:
                            raise BrokenPipeError("PipeWire exited while producer input open")
                        if time.monotonic() >= min(idle_deadline, self._deadline):
                            raise TimeoutError("producer idle deadline")
                        self._condition.wait(_POLL_SECONDS)
                    if not self._pending:
                        break
                    segment = self._active = self._pending.popleft()
                frame_block = max(1, _BLOCK_BYTES // (self.channels * 2))
                remaining_frames = segment.frames
                while remaining_frames:
                    count = min(frame_block, remaining_frames)
                    data = segment.reader.readframes(count)
                    if len(data) != count * self.channels * 2:
                        raise wave.Error("PCM changed or truncated after acceptance")
                    self._write(data)
                    remaining_frames -= count
                segment.close()
                with self._condition:
                    self._active = None
                    self._condition.notify_all()
            if self._cancel.is_set():
                raise InterruptedError("playback cancelled")
            if self.process.poll() is not None:
                raise BrokenPipeError("PipeWire exited before stdin EOF")
            self.process.stdin.close()
            deadline = min(self._deadline, time.monotonic() + _DRAIN_SECONDS)
            while self.process.poll() is None:
                if self._cancel.wait(_POLL_SECONDS):
                    raise InterruptedError("playback cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError("PipeWire drain deadline")
            returncode = self.process.wait(timeout=0.25)
            if returncode != 0:
                raise BrokenPipeError("PipeWire drain failed")
            completed = True
        except Exception as exc:
            with self._condition:
                if self._state == "playing":
                    self._state = "failed"
                    self._failure = ("stream_timeout" if isinstance(exc, TimeoutError)
                                     else "stream_io_failed")
        finally:
            if not completed:
                _terminate(self.process)
            if self.process.stdin is not None:
                self.process.stdin.close()
            with self._condition:
                segments = list(self._pending)
                if self._active is not None:
                    segments.append(self._active)
                self._pending.clear()
                self._active = None
                for segment in segments:
                    segment.close()
                if completed and self._state == "playing":
                    self._state = "completed"
                self._finalized.set()
                self._condition.notify_all()


def _create_stream_playback(args):
    capability = "audio.start_stream_playback"
    owner = physical._validate_owner(args["media_owner"], capability)
    role = args.get("role", "audio.primary_output")
    resolved, device = physical._resolve_role(role, "audio_output", capability)
    settings = resolved.get("settings") or {}
    volume = physical._bounded_volume(args.get("volume_percent"), settings)
    lead_in_ms = physical._playback_lead_in(settings, device)
    first = _open_segment(args["media_id"], owner, capability)
    process = None
    try:
        physical._cleanup_playbacks()
        with physical._LOCK:
            if any(record.process.poll() is None for record in physical._PLAYBACKS.values()):
                raise HardwareError("已有实体播放占用音频输出", capability=capability)
            command = [
                # PipeWire 1.0.5 selects setup_pipe()/raw stdin via filename
                # "-"; it has no --raw option. Explicit format/rate/channels
                # configure the pipe without a WAV header or libsndfile.
                "pw-play", "--target", device["endpoint"], "--format", "s16",
                "--rate", str(first.sample_rate), "--channels", str(first.channels),
                "--volume", f"{volume / 100:.2f}", "-",
            ]
            process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       bufsize=0)
            if process.stdin is None:
                raise OSError("missing PCM stdin")
            os.set_blocking(process.stdin.fileno(), False)
            playback_id = f"playback_{secrets.token_urlsafe(18)}"
            record = StreamPlayback(playback_id, owner, first, role, device.get("device_id", ""),
                                    process, lead_in_ms, args["stream_key"])
            physical._PLAYBACKS[playback_id] = record
            try:
                record.start()
            except BaseException:
                physical._PLAYBACKS.pop(playback_id, None)
                raise
        return record.public()
    except BaseException as exc:
        first.close()
        if process is not None:
            _terminate(process)
            if process.stdin is not None:
                process.stdin.close()
        if isinstance(exc, OSError):
            raise HardwareError("无法启动PipeWire连续播放", capability=capability) from exc
        raise


def start_stream_playback(args):
    """A bounded receipt makes an uncertain start retry return the same child."""
    capability = "audio.start_stream_playback"
    owner = physical._validate_owner(args["media_owner"], capability)
    stream_key = args.get("stream_key")
    if not isinstance(stream_key, str) or not _STREAM_KEY.fullmatch(stream_key):
        raise SchemaValidationError("stream_key格式无效", capability=capability)
    resolve_only = args.get("resolve_only", False)
    if not isinstance(resolve_only, bool):
        raise SchemaValidationError("resolve_only必须是boolean", capability=capability)
    role = args.get("role", "audio.primary_output")
    volume = args.get("volume_percent")
    if volume is not None:
        physical._bounded_volume(volume, {})
    identity = (args["media_id"], role, volume)
    key = (owner, stream_key)
    with _START_LOCK:
        with physical._LOCK:
            now = time.monotonic()
            for old_key, (_, old_record, created) in list(_RECEIPTS.items()):
                if now - created >= _RECEIPT_SECONDS and old_record.public()["state"] != "playing":
                    _RECEIPTS.pop(old_key, None)
                    physical._PLAYBACKS.pop(old_record.playback_id, None)
            receipt = _RECEIPTS.get(key)
            if receipt is not None:
                previous_identity, record, _ = receipt
                if identity != previous_identity:
                    raise SchemaValidationError("stream_key与先前播放参数冲突", capability=capability)
                return record.public()
            if resolve_only:
                raise ResourceNotFoundError("连续播放回执不存在或已经过期", capability=capability)
            while len(_RECEIPTS) >= _MAX_RECEIPTS:
                ended = next((old_key for old_key, (_, old_record, _) in _RECEIPTS.items()
                              if old_record.public()["state"] != "playing"), None)
                if ended is None:
                    raise HardwareError("连续播放回执容量已满", capability=capability)
                _, old_record, _ = _RECEIPTS.pop(ended)
                physical._PLAYBACKS.pop(old_record.playback_id, None)
        result = _create_stream_playback(args)
        with physical._LOCK:
            record = physical._PLAYBACKS[result["playback_id"]]
            _RECEIPTS[key] = (identity, record, time.monotonic())
        return result


def _require_stream(playback_id):
    record = physical._require_playback(playback_id)
    if not getattr(record, "streaming", False):
        raise SchemaValidationError("播放句柄不是连续PCM流", capability="audio")
    return record


def append_stream_playback(args):
    return _require_stream(args["playback_id"]).add(
        args["media_id"], args["media_owner"], args["segment_index"])


def finish_stream_playback(args):
    return _require_stream(args["playback_id"]).finish(args["media_owner"])
