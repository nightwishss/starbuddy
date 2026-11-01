"""Physical PipeWire audio adapters using logical roles and shared media IDs."""

from __future__ import annotations

import math
import os
import re
import select
import secrets
import signal
import subprocess
import threading
import time
import wave
from array import array
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from device_management import HardwareProfileManager
from media_service import MediaClient, MediaClientError
from runtime.errors import HardwareError, ResourceNotFoundError, SchemaValidationError

_RUNTIME_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
_WORK_ROOT = _RUNTIME_DIR / "vq-audio" / "work"
_OWNER = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_MEDIA_ID = re.compile(r"^(?:audio|generated)_[A-Za-z0-9_-]{20,64}$")
_PLAYBACK_ID = re.compile(r"^playback_[A-Za-z0-9_-]{20,64}$")
_MAX_PLAYBACK_SECONDS = 120.0
_SILENCE_DBFS = -48.0
_PCM_CHUNK_MS = 20
_ECHO_TAIL_GUARD_MS = 180


@dataclass
class Playback:
    playback_id: str
    owner: str
    media_id: str
    role: str
    device_id: str
    duration_seconds: float
    started_at: float
    process: subprocess.Popen
    stopped: bool = False
    temporary_path: Path | None = None
    lead_in_ms: int = 0


_PLAYBACKS: dict[str, Playback] = {}
_LOCK = threading.RLock()
_PLAYBACK_START_LOCK = threading.RLock()


def _media_client():
    return MediaClient()


def _validate_owner(owner: str, capability: str) -> str:
    if not isinstance(owner, str) or not _OWNER.fullmatch(owner):
        raise SchemaValidationError("media_owner格式无效", capability=capability)
    return owner


def _resolve_role(role: str, kind: str, capability: str) -> tuple[dict, dict]:
    if not isinstance(role, str) or not role.startswith("audio."):
        raise SchemaValidationError("音频逻辑角色无效", capability=capability)
    resolved = HardwareProfileManager().resolve(role)
    if resolved.get("kind") != kind:
        raise SchemaValidationError("音频角色方向不匹配", capability=capability)
    device = resolved.get("device") or {}
    if not isinstance(device.get("endpoint"), str) or not device["endpoint"]:
        raise ResourceNotFoundError("音频设备端点不可用", capability=capability)
    return resolved, device


def _bounded_volume(value, settings: dict) -> int:
    volume = settings.get("volume_percent", 100) if value is None else value
    if isinstance(volume, bool) or not isinstance(volume, int) or not 0 <= volume <= 100:
        raise SchemaValidationError("volume_percent必须在0..100之间", capability="audio")
    return volume


def interaction_status(args=None):
    """Report the effective product audio policy without exposing PipeWire endpoints."""
    module_loaded = False
    server_available = False
    try:
        result = subprocess.run(
            ["pactl", "list", "modules", "short"], capture_output=True,
            text=True, timeout=3, check=False,
        )
        server_available = result.returncode == 0
        module_loaded = any(
            "module-echo-cancel" in line for line in result.stdout.splitlines()
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    roles = HardwareProfileManager().resolve_all().get("roles") or {}
    aec_input = (roles.get("audio.echo_cancelled_input") or {}).get("status")
    aec_output = (roles.get("audio.echo_reference_output") or {}).get("status")
    available = module_loaded and aec_input == "resolved" and aec_output == "resolved"
    return {
        "mode": "half_duplex",
        "trigger": "explicit",
        "playback_interruption": True,
        "echo_guard_ms": _ECHO_TAIL_GUARD_MS,
        "aec": {
            "enabled": False,
            "module_loaded": module_loaded,
            "server_available": server_available,
            "capture_role_bound": False,
            "available": available,
            "input_role": "audio.echo_cancelled_input",
            "output_role": "audio.echo_reference_output",
            "input_role_status": aec_input or "missing",
            "output_role_status": aec_output or "missing",
            "fallback_input_role": "audio.primary_input",
            "fallback_output_role": "audio.primary_output",
        },
        "wake_word": {
            "enabled": False,
            "background_capture": False,
        },
    }


def _run_capture(target: str, output: Path, duration_seconds: float,
                 sample_rate: int, channels: int, volume: int) -> None:
    command = [
        "pw-record", "--target", target, "--rate", str(sample_rate),
        "--channels", str(channels), "--format", "s16",
        "--volume", f"{volume / 100:.2f}", str(output),
    ]
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True,
        )
        interrupted = False
        try:
            returncode = process.wait(timeout=duration_seconds)
        except subprocess.TimeoutExpired:
            interrupted = True
            process.send_signal(signal.SIGINT)
            try:
                returncode = process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
                returncode = process.returncode
    except OSError as exc:
        raise HardwareError("无法启动PipeWire录音", capability="audio.capture_clip") from exc
    # PipeWire 1.0.5 returns 1 after a requested SIGINT even though it closes
    # the WAV correctly. An early returncode=1 is still a hardware failure.
    if returncode not in {0, -signal.SIGINT, 130} and not (interrupted and returncode == 1):
        output.unlink(missing_ok=True)
        raise HardwareError("实体麦克风录音失败", capability="audio.capture_clip",
                            details={"returncode": returncode})


def _wav_metrics(path: Path) -> dict:
    try:
        with wave.open(str(path), "rb") as stream:
            sample_rate = stream.getframerate()
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            frames = stream.getnframes()
            if sample_width != 2 or sample_rate <= 0 or channels not in {1, 2} or frames <= 0:
                raise wave.Error("unsupported recording format")
            samples = array("h")
            samples.frombytes(stream.readframes(frames))
    except (OSError, EOFError, wave.Error) as exc:
        raise HardwareError("实体录音WAV无效", capability="audio") from exc
    if os.sys.byteorder != "little":
        samples.byteswap()
    peak = max((abs(value) for value in samples), default=0)
    rms = math.sqrt(sum(value * value for value in samples) / max(1, len(samples)))
    rms_dbfs = 20 * math.log10(max(rms, 1.0) / 32768)
    peak_dbfs = 20 * math.log10(max(peak, 1) / 32768)
    return {
        "sample_rate": sample_rate, "channels": channels, "sample_width": sample_width,
        "frames": frames, "duration_seconds": round(frames / sample_rate, 3),
        "rms_dbfs": round(rms_dbfs, 2), "peak_dbfs": round(peak_dbfs, 2),
        "silent": rms_dbfs < _SILENCE_DBFS,
    }


def _pcm_dbfs(data: bytes) -> float:
    samples = array("h")
    samples.frombytes(data[:len(data) - len(data) % 2])
    if os.sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return -90.0
    rms = math.sqrt(sum(value * value for value in samples) / len(samples))
    return max(-90.0, 20 * math.log10(max(rms, 1.0) / 32768))


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1)


def _capture_turn_pcm(target: str, *, sample_rate: int, channels: int, volume: int,
                      max_wait_ms: int, max_turn_ms: int, end_silence_ms: int,
                      pre_roll_ms: int, calibration_ms: int) -> tuple[bytes | None, dict]:
    command = [
        "pw-record", "--target", target, "--rate", str(sample_rate),
        "--channels", str(channels), "--format", "s16",
        "--volume", f"{volume / 100:.2f}", "-",
    ]
    chunk_bytes = sample_rate * channels * 2 * _PCM_CHUNK_MS // 1000
    pre_roll = deque(maxlen=max(1, pre_roll_ms // _PCM_CHUNK_MS))
    calibration_chunks = max(1, calibration_ms // _PCM_CHUNK_MS)
    end_silence_chunks = max(1, end_silence_ms // _PCM_CHUNK_MS)
    max_wait_chunks = max(1, max_wait_ms // _PCM_CHUNK_MS)
    max_turn_chunks = max(1, max_turn_ms // _PCM_CHUNK_MS)
    noise_values: list[float] = []
    onset_window: deque[bool] = deque(maxlen=5)
    captured: list[bytes] = []
    speech_started = False
    speech_start_chunk = None
    silence_chunks = 0
    total_chunks = 0
    end_reason = "no_speech"
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise HardwareError("无法启动PipeWire语音采集", capability="audio.capture_turn") from exc
    try:
        if process.stdout is None:
            raise HardwareError("PipeWire语音采集管道无效", capability="audio.capture_turn")
        deadline = time.monotonic() + (calibration_ms + max_wait_ms + max_turn_ms) / 1000 + 3
        while total_chunks < calibration_chunks + max_wait_chunks + max_turn_chunks:
            if time.monotonic() >= deadline:
                raise HardwareError("实体麦克风语音采集超时", capability="audio.capture_turn")
            ready, _, _ = select.select([process.stdout], [], [], 1.0)
            if not ready:
                if process.poll() is not None:
                    raise HardwareError("实体麦克风在语音采集中断开", capability="audio.capture_turn")
                continue
            data = process.stdout.read(chunk_bytes)
            if len(data) != chunk_bytes:
                if process.poll() is not None:
                    raise HardwareError("实体麦克风在语音采集中断开", capability="audio.capture_turn")
                continue
            total_chunks += 1
            level = _pcm_dbfs(data)
            if len(noise_values) < calibration_chunks:
                noise_values.append(level)
                pre_roll.append(data)
                continue
            ordered_noise = sorted(noise_values)
            noise_floor = ordered_noise[len(ordered_noise) // 2]
            threshold = max(-50.0, min(-26.0, noise_floor + 12.0))
            active = level >= threshold
            if not speech_started:
                pre_roll.append(data)
                onset_window.append(active)
                if sum(onset_window) >= 3:
                    speech_started = True
                    speech_start_chunk = total_chunks
                    captured.extend(pre_roll)
                    silence_chunks = 0
                elif total_chunks >= calibration_chunks + max_wait_chunks:
                    end_reason = "no_speech"
                    break
                continue
            captured.append(data)
            silence_chunks = 0 if active else silence_chunks + 1
            speech_chunks = total_chunks - (speech_start_chunk or total_chunks) + 1
            if speech_chunks >= max_turn_chunks:
                end_reason = "max_duration"
                break
            if speech_chunks >= 15 and silence_chunks >= end_silence_chunks:
                end_reason = "end_silence"
                break
    finally:
        _stop_process(process)
    noise_floor = sorted(noise_values)[len(noise_values) // 2] if noise_values else -90.0
    threshold = max(-50.0, min(-26.0, noise_floor + 12.0))
    metadata = {
        "detected": speech_started,
        "end_reason": end_reason,
        "noise_floor_dbfs": round(noise_floor, 2),
        "threshold_dbfs": round(threshold, 2),
        "pre_roll_ms": pre_roll_ms,
        "end_silence_ms": end_silence_ms,
        "capture_ms": total_chunks * _PCM_CHUNK_MS,
    }
    return (b"".join(captured) if speech_started else None), metadata


def capture_turn(args):
    capability = "audio.capture_turn"
    owner = _validate_owner(args["media_owner"], capability)
    role = args.get("role", "audio.primary_input")
    resolved, device = _resolve_role(role, "audio_input", capability)
    sample_rate = args.get("sample_rate", 16000)
    channels = args.get("channels", 1)
    volume = _bounded_volume(args.get("volume_percent"), resolved.get("settings") or {})
    pcm, turn = _capture_turn_pcm(
        device["endpoint"], sample_rate=sample_rate, channels=channels, volume=volume,
        max_wait_ms=args.get("max_wait_ms", 8000),
        max_turn_ms=args.get("max_turn_ms", 15000),
        end_silence_ms=args.get("end_silence_ms", 700),
        pre_roll_ms=args.get("pre_roll_ms", 300),
        calibration_ms=args.get("calibration_ms", 400),
    )
    base = {
        "detected": bool(pcm), "role": role,
        "device_id": device.get("device_id", ""), "volume_percent": volume,
        "turn": turn,
    }
    if not pcm:
        return base
    _WORK_ROOT.mkdir(parents=True, exist_ok=True)
    _WORK_ROOT.chmod(0o700)
    output = _WORK_ROOT / f"turn-{secrets.token_urlsafe(18)}.wav"
    try:
        with wave.open(str(output), "wb") as stream:
            stream.setframerate(sample_rate)
            stream.setnchannels(channels)
            stream.setsampwidth(2)
            stream.writeframes(pcm)
        output.chmod(0o600)
        signal_metrics = _wav_metrics(output)
        try:
            media = _media_client().create_from_path(owner, "audio/wav", output)
        except (OSError, MediaClientError) as exc:
            raise HardwareError("Media Service暂时不可用", capability=capability) from exc
    finally:
        output.unlink(missing_ok=True)
    return {**base, **media, "signal": signal_metrics}


def capture_clip(args):
    capability = "audio.capture_clip"
    owner = _validate_owner(args["media_owner"], capability)
    role = args.get("role", "audio.primary_input")
    resolved, device = _resolve_role(role, "audio_input", capability)
    duration_ms = args.get("duration_ms", 4000)
    sample_rate = args.get("sample_rate", 16000)
    channels = args.get("channels", 1)
    volume = _bounded_volume(args.get("volume_percent"), resolved.get("settings") or {})
    _WORK_ROOT.mkdir(parents=True, exist_ok=True)
    _WORK_ROOT.chmod(0o700)
    output = _WORK_ROOT / f"capture-{secrets.token_urlsafe(18)}.wav"
    try:
        _run_capture(device["endpoint"], output, duration_ms / 1000,
                     sample_rate, channels, volume)
        metrics = _wav_metrics(output)
        try:
            media = _media_client().create_from_path(owner, "audio/wav", output)
        except (OSError, MediaClientError) as exc:
            raise HardwareError("Media Service暂时不可用", capability=capability) from exc
    finally:
        output.unlink(missing_ok=True)
    return {
        **media, "role": role, "device_id": device.get("device_id", ""),
        "volume_percent": volume, "signal": metrics,
    }


def _shared_wav(media_id: str, owner: str, capability: str) -> tuple[Path, dict]:
    if not isinstance(media_id, str) or not _MEDIA_ID.fullmatch(media_id):
        raise SchemaValidationError("media_id格式无效", capability=capability)
    _validate_owner(owner, capability)
    try:
        client = _media_client()
        info = client.call("info", owner=owner, media_id=media_id)
        resolved = client.call("resolve", owner=owner, media_id=media_id)
        if info.get("mime_type") != "audio/wav":
            raise SchemaValidationError("实体播放只接受WAV媒体", capability=capability)
        return Path(resolved["path"]), info
    except SchemaValidationError:
        raise
    except MediaClientError as exc:
        raise ResourceNotFoundError("音频媒体不存在、过期或无权访问", capability=capability) from exc
    except (OSError, KeyError) as exc:
        raise HardwareError("Media Service暂时不可用", capability=capability) from exc


def _cleanup_playbacks() -> None:
    cutoff = time.time() - 600
    with _LOCK:
        for playback_id, record in list(_PLAYBACKS.items()):
            if record.process.poll() is not None:
                _release_playback_file(record)
                if record.started_at < cutoff:
                    _PLAYBACKS.pop(playback_id, None)


def _playback_format(path: Path) -> dict:
    """Validate PCM and duration without scanning the entire waveform for RMS."""
    try:
        with wave.open(str(path), "rb") as stream:
            rate, channels, width, frames = (
                stream.getframerate(), stream.getnchannels(),
                stream.getsampwidth(), stream.getnframes(),
            )
            if width != 2 or rate <= 0 or channels not in {1, 2} or frames <= 0:
                raise wave.Error("unsupported PCM")
            duration = frames / rate
            if duration > _MAX_PLAYBACK_SECONDS:
                raise SchemaValidationError("播放音频超过120秒限制", capability="audio.start_playback")
            stream.setpos(frames - 1)
            if len(stream.readframes(1)) != channels * width:
                raise wave.Error("truncated PCM")
            return {"duration_seconds": round(duration, 3), "sample_rate": rate,
                    "channels": channels, "sample_width": width, "frames": frames}
    except (OSError, EOFError, wave.Error) as exc:
        raise HardwareError("实体播放WAV无效", capability="audio.start_playback") from exc


def _playback_lead_in(settings: dict, device: dict) -> int:
    # Some HDMI monitor amplifiers mute the initial PCM even with PipeWire
    # suspend disabled. Keep a bounded PCM lead-in in the same playback stream;
    # do not create a persistent silent player or another device owner.
    default = 250 if "hdmi" in device.get("name", "").lower() else 0
    value = settings.get("playback_lead_in_ms", default)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 800:
        raise SchemaValidationError("playback_lead_in_ms必须在0..800之间", capability="audio.start_playback")
    return value


def _prepend_playback_silence(path: Path, lead_in_ms: int) -> Path:
    _WORK_ROOT.mkdir(parents=True, exist_ok=True)
    _WORK_ROOT.chmod(0o700)
    target = _WORK_ROOT / f"playback-{secrets.token_urlsafe(18)}.wav"
    try:
        # Private, bounded temporary copy: preserve every original PCM byte.
        with target.open("xb") as raw:
            os.chmod(target, 0o600)
            with wave.open(str(path), "rb") as source, wave.open(raw, "wb") as output:
                output.setparams(source.getparams())
                lead_frames = source.getframerate() * lead_in_ms // 1000
                output.writeframesraw(bytes(lead_frames * source.getnchannels() * source.getsampwidth()))
                while True:
                    block = source.readframes(16384)
                    if not block:
                        break
                    output.writeframesraw(block)
        return target
    except (OSError, EOFError, wave.Error) as exc:
        target.unlink(missing_ok=True)
        raise HardwareError("无法准备实体播放音频", capability="audio.start_playback") from exc


def _release_playback_file(record: Playback) -> None:
    if record.temporary_path is not None:
        record.temporary_path.unlink(missing_ok=True)
        record.temporary_path = None


def _start_playback(args):
    capability = "audio.start_playback"
    with _LOCK:
        if any(getattr(record, "streaming", False) and record.process.poll() is None
               for record in _PLAYBACKS.values()):
            raise HardwareError("连续播放正在占用音频输出", capability=capability)
    owner = _validate_owner(args["media_owner"], capability)
    role = args.get("role", "audio.primary_output")
    resolved, device = _resolve_role(role, "audio_output", capability)
    settings = resolved.get("settings") or {}
    volume = _bounded_volume(args.get("volume_percent"), settings)
    path, info = _shared_wav(args["media_id"], owner, capability)
    metrics = _playback_format(path)
    lead_in_ms = _playback_lead_in(settings, device)
    _cleanup_playbacks()
    playback_path = _prepend_playback_silence(path, lead_in_ms) if lead_in_ms else path
    command = [
        "pw-play", "--target", device["endpoint"],
        "--volume", f"{volume / 100:.2f}", str(playback_path),
    ]
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        if playback_path != path:
            playback_path.unlink(missing_ok=True)
        raise HardwareError("无法启动PipeWire播放", capability=capability) from exc
    playback_id = f"playback_{secrets.token_urlsafe(18)}"
    record = Playback(
        playback_id, owner, info["media_id"], role, device.get("device_id", ""),
        metrics["duration_seconds"] + lead_in_ms / 1000, time.time(), process,
        temporary_path=playback_path if playback_path != path else None,
        lead_in_ms=lead_in_ms,
    )
    with _LOCK:
        _PLAYBACKS[playback_id] = record
    return _playback_public(record)


def start_playback(args):
    with _PLAYBACK_START_LOCK:
        return _start_playback(args)


def _require_playback(playback_id: str) -> Playback:
    if not isinstance(playback_id, str) or not _PLAYBACK_ID.fullmatch(playback_id):
        raise SchemaValidationError("playback_id格式无效", capability="audio")
    with _LOCK:
        record = _PLAYBACKS.get(playback_id)
    if record is None:
        raise ResourceNotFoundError("播放任务不存在或已经关闭", capability="audio")
    return record


def _playback_public(record: Playback) -> dict:
    if getattr(record, "streaming", False):
        return record.public()
    returncode = record.process.poll()
    if returncode is not None:
        _release_playback_file(record)
    state = "stopped" if record.stopped else "playing" if returncode is None else (
        "completed" if returncode == 0 else "failed")
    value = {
        "playback_id": record.playback_id, "media_id": record.media_id,
        "role": record.role, "device_id": record.device_id,
        "state": state, "duration_seconds": record.duration_seconds,
        "elapsed_seconds": round(max(0.0, time.time() - record.started_at), 3),
        "lead_in_ms": record.lead_in_ms,
    }
    if returncode is not None:
        value["returncode"] = returncode
    return value


def playback_status(args):
    return _playback_public(_require_playback(args["playback_id"]))


def stop_playback(args):
    record = _require_playback(args["playback_id"])
    if getattr(record, "streaming", False):
        result = record.stop()
        with _LOCK:
            _PLAYBACKS.pop(record.playback_id, None)
        return result
    if record.process.poll() is None:
        record.process.send_signal(signal.SIGINT)
        try:
            record.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            record.process.kill()
            record.process.wait(timeout=1)
    record.stopped = True
    result = _playback_public(record)
    with _LOCK:
        _PLAYBACKS.pop(record.playback_id, None)
    return result


def start_stream_playback(args):
    from .streaming import start_stream_playback as start
    with _PLAYBACK_START_LOCK:
        return start(args)


def append_stream_playback(args):
    from .streaming import append_stream_playback as append
    return append(args)


def finish_stream_playback(args):
    from .streaming import finish_stream_playback as finish
    return finish(args)
