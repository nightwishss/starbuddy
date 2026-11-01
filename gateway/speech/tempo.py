"""Bounded, pitch-preserving slow playback after the configured TTS backend.

Keep normal synthesis (including QNN) unchanged. No extra work at speed=1.
"""
import os
import subprocess
import tempfile
from pathlib import Path


def validate_speed(speed):
    if type(speed) not in (int, float) or not 0.6 <= speed <= 1.0:
        raise ValueError("speech speed must be 0.6..1.0")
    return float(speed)


def slow_wave(path, speed):
    speed = validate_speed(speed)
    if speed == 1:
        return
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="runtime-tts-tempo-", suffix=".wav", dir=path.parent)
    os.close(fd)
    try:
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-threads", "1", "-i", str(path), "-filter_threads", "1",
            "-af", f"atempo={speed}", "-c:a", "pcm_s16le", temporary,
        ], check=True, timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
