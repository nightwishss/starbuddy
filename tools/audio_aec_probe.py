#!/usr/bin/env python3
"""Bounded far-end acoustic echo probe for the logical USB audio roles.

The default invocation is inventory-only. ``--execute`` plays a deterministic,
low-volume speech-band stimulus, records the physical input, repeats through a
temporary WebRTC echo-cancel module, prints metadata metrics, and deletes all
audio artifacts. It does not enable AEC for the product.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import wave
from array import array
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from device_management import HardwareProfileManager


SAMPLE_RATE = 48_000
CAPTURE_LEAD_SECONDS = 0.7
CAPTURE_TAIL_SECONDS = 0.5
SOURCE_NAME = "vq_aec_probe_source"
SINK_NAME = "vq_aec_probe_sink"


class ProbeError(RuntimeError):
    pass


def _run(command: list[str], *, timeout=5, check=True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=check,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProbeError(f"command failed: {command[0]}") from exc


def _logical_audio() -> tuple[dict, dict]:
    manager = HardwareProfileManager()
    source = manager.resolve("audio.primary_input")
    sink = manager.resolve("audio.primary_output")
    if source.get("status") != "resolved" or sink.get("status") != "resolved":
        raise ProbeError("logical input/output is not resolved")
    return source["device"], sink["device"]


def _active_streams() -> dict[str, int]:
    result = {}
    for name, command in {
        "capture": ["pactl", "list", "source-outputs", "short"],
        "playback": ["pactl", "list", "sink-inputs", "short"],
    }.items():
        completed = _run(command, check=False)
        result[name] = sum(bool(line.strip()) for line in completed.stdout.splitlines())
    return result


def _write_stimulus(path: Path, duration_seconds: float) -> None:
    """Create deterministic, speech-band, amplitude-ramped mono PCM."""
    count = int(SAMPLE_RATE * duration_seconds)
    state = 0x51A7C0DE
    filtered = 0.0
    samples = array("h")
    ramp_frames = int(SAMPLE_RATE * 0.08)
    for index in range(count):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        noise = ((state >> 8) / 0xFFFFFF) * 2.0 - 1.0
        # Low-pass pseudo-noise plus speech-like tones gives WebRTC AEC a
        # repeatable far-end reference without storing spoken content.
        filtered = 0.82 * filtered + 0.18 * noise
        tone = 0.22 * math.sin(2 * math.pi * 430 * index / SAMPLE_RATE)
        tone += 0.12 * math.sin(2 * math.pi * 870 * index / SAMPLE_RATE)
        ramp = min(1.0, index / max(1, ramp_frames), (count - index) / max(1, ramp_frames))
        value = max(-1.0, min(1.0, (0.48 * filtered + tone) * ramp))
        samples.append(int(value * 32767))
    with wave.open(str(path), "wb") as stream:
        stream.setframerate(SAMPLE_RATE)
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.writeframes(samples.tobytes())


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1)


def _capture_playback(source: str, sink: str, stimulus: Path, output: Path,
                      duration_seconds: float, volume_percent: int) -> None:
    record = subprocess.Popen([
        "pw-record", "--target", source, "--rate", str(SAMPLE_RATE),
        "--channels", "1", "--format", "s16", str(output),
    ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        time.sleep(CAPTURE_LEAD_SECONDS)
        playback = _run([
            "pw-play", "--target", sink, "--volume", f"{volume_percent / 100:.2f}",
            str(stimulus),
        ], timeout=duration_seconds + 4, check=False)
        if playback.returncode != 0:
            raise ProbeError("test stimulus playback failed")
        time.sleep(CAPTURE_TAIL_SECONDS)
    finally:
        _stop(record)
    if not output.is_file() or output.stat().st_size < 44:
        raise ProbeError("test capture is missing or invalid")


def _samples(path: Path) -> tuple[int, list[int]]:
    try:
        with wave.open(str(path), "rb") as stream:
            rate = stream.getframerate()
            channels = stream.getnchannels()
            if stream.getsampwidth() != 2 or channels not in {1, 2}:
                raise ProbeError("unsupported capture WAV")
            raw = array("h")
            raw.frombytes(stream.readframes(stream.getnframes()))
    except (OSError, EOFError, wave.Error) as exc:
        raise ProbeError("invalid capture WAV") from exc
    if os.sys.byteorder != "little":
        raw.byteswap()
    if channels == 2:
        values = [(raw[index] + raw[index + 1]) // 2 for index in range(0, len(raw), 2)]
    else:
        values = list(raw)
    return rate, values


def _dbfs(values: list[int]) -> tuple[float, float]:
    if not values:
        return -90.0, -90.0
    rms = math.sqrt(sum(value * value for value in values) / len(values))
    peak = max(abs(value) for value in values)
    return (
        max(-90.0, 20 * math.log10(max(rms, 1.0) / 32768)),
        max(-90.0, 20 * math.log10(max(peak, 1.0) / 32768)),
    )


def _metrics(path: Path, duration_seconds: float) -> dict:
    rate, values = _samples(path)
    noise = values[int(rate * 0.1):int(rate * 0.55)]
    far = values[
        int(rate * CAPTURE_LEAD_SECONDS):
        int(rate * (CAPTURE_LEAD_SECONDS + duration_seconds))
    ]
    noise_rms, noise_peak = _dbfs(noise)
    far_rms, far_peak = _dbfs(far)
    return {
        "sample_rate": rate,
        "duration_seconds": round(len(values) / rate, 3),
        "noise_rms_dbfs": round(noise_rms, 2),
        "noise_peak_dbfs": round(noise_peak, 2),
        "far_end_rms_dbfs": round(far_rms, 2),
        "far_end_peak_dbfs": round(far_peak, 2),
        "far_end_over_noise_db": round(far_rms - noise_rms, 2),
    }


def _load_aec(source: str, sink: str) -> str:
    completed = _run([
        "pactl", "load-module", "module-echo-cancel", "aec_method=webrtc",
        f"source_master={source}", f"sink_master={sink}",
        f"source_name={SOURCE_NAME}", f"sink_name={SINK_NAME}",
        "source_properties=device.description=VQ_AEC_Probe_Source",
        "sink_properties=device.description=VQ_AEC_Probe_Sink",
    ])
    module_id = completed.stdout.strip()
    if not module_id.isdigit():
        raise ProbeError("echo-cancel module did not return an id")
    return module_id


def _unload_aec(module_id: str | None) -> None:
    if module_id:
        _run(["pactl", "unload-module", module_id], check=False)


def probe(*, execute: bool, duration_seconds: float, volume_percent: int,
          allow_busy: bool = False) -> dict:
    if not 1.0 <= duration_seconds <= 5.0:
        raise ProbeError("duration must be between 1 and 5 seconds")
    if not 1 <= volume_percent <= 40:
        raise ProbeError("volume must be between 1 and 40 percent")
    source, sink = _logical_audio()
    streams = _active_streams()
    inventory = {
        "execute": execute,
        "input_device_id": source.get("device_id"),
        "output_device_id": sink.get("device_id"),
        "duration_seconds": duration_seconds,
        "volume_percent": volume_percent,
        "active_streams": streams,
        "artifacts_retained": False,
    }
    if not execute:
        return {**inventory, "status": "dry_run"}
    if not allow_busy and any(streams.values()):
        raise ProbeError("audio graph is busy; stop capture/playback before probing")

    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    lock_path = runtime_dir / "vq-aec-probe.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProbeError("another AEC probe is running") from exc
        module_id = None
        with tempfile.TemporaryDirectory(prefix="vq-aec-probe-", dir=runtime_dir) as temporary:
            root = Path(temporary)
            stimulus = root / "stimulus.wav"
            baseline = root / "baseline.wav"
            cancelled = root / "aec.wav"
            _write_stimulus(stimulus, duration_seconds)
            try:
                _capture_playback(
                    source["endpoint"], sink["endpoint"], stimulus, baseline,
                    duration_seconds, volume_percent,
                )
                baseline_metrics = _metrics(baseline, duration_seconds)
                module_id = _load_aec(source["endpoint"], sink["endpoint"])
                time.sleep(1.0)
                _capture_playback(
                    SOURCE_NAME, SINK_NAME, stimulus, cancelled,
                    duration_seconds, volume_percent,
                )
                aec_metrics = _metrics(cancelled, duration_seconds)
            finally:
                _unload_aec(module_id)
        attenuation = baseline_metrics["far_end_rms_dbfs"] - aec_metrics["far_end_rms_dbfs"]
        return {
            **inventory,
            "status": "measured",
            "baseline": baseline_metrics,
            "aec": aec_metrics,
            "far_end_attenuation_db": round(attenuation, 2),
            "preliminary_far_end_pass": attenuation >= 6.0,
            "near_end_speech_tested": False,
            "product_aec_enabled": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="play the bounded test stimulus")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--volume", type=int, default=20)
    parser.add_argument("--allow-busy", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    try:
        result = probe(
            execute=arguments.execute, duration_seconds=arguments.duration,
            volume_percent=arguments.volume, allow_busy=arguments.allow_busy,
        )
    except ProbeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
