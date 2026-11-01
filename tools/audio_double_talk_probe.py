#!/usr/bin/env python3
"""Interactive near-end and double-talk gate for optional AEC logical roles."""

from __future__ import annotations

import argparse
import difflib
import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_service import EchoCancelService
from audio_service.echo_cancel import AEC_SINK_NAME, AEC_SOURCE_NAME
from media_service import MediaClient
from runtime.client import RuntimeClient
from tools import audio_aec_probe


REFERENCE_TEXT = "星伴今天和我一起观察蓝色杯子"
OWNER = "service:aec-double-talk-probe"
SAMPLE_RATE = 16_000
LEAD_SECONDS = 0.5


class DoubleTalkError(RuntimeError):
    pass


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1)


def _record(source: str, output: Path, duration: float, *, sink=None,
            stimulus=None, volume_percent=15) -> None:
    record = subprocess.Popen([
        "pw-record", "--target", source, "--rate", str(SAMPLE_RATE),
        "--channels", "1", "--format", "s16", str(output),
    ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        time.sleep(LEAD_SECONDS)
        print(f"现在朗读：{REFERENCE_TEXT}", flush=True)
        if sink and stimulus:
            completed = subprocess.run([
                "pw-play", "--target", sink, "--volume", f"{volume_percent / 100:.2f}",
                str(stimulus),
            ], capture_output=True, timeout=duration + 4, check=False)
            if completed.returncode != 0:
                raise DoubleTalkError("far-end playback failed")
        else:
            time.sleep(duration)
    finally:
        _stop(record)
    if not output.is_file() or output.stat().st_size < 44:
        raise DoubleTalkError("near-end capture is missing")


def _transcribe(path: Path, *, media=None, runtime=None) -> str:
    media = media or MediaClient()
    runtime = runtime or RuntimeClient()
    item = media.create_from_path(OWNER, "audio/wav", path, ttl_seconds=120)
    try:
        response = runtime.call(
            "speech.transcribe", {"media_id": item["media_id"], "media_owner": OWNER},
            timeout_ms=30_000,
        )
        if not response.get("ok"):
            raise DoubleTalkError("local STT failed")
        return str((response.get("result") or {}).get("text", "")).strip()
    finally:
        media.call("release", owner=OWNER, media_id=item["media_id"])


def _similarity(text: str) -> float:
    compact = "".join(character for character in text if character.isalnum())
    return round(difflib.SequenceMatcher(None, REFERENCE_TEXT, compact).ratio(), 3)


def _decision(physical: float, aec_near: float, double_talk: float) -> dict:
    near_preserved = aec_near >= max(0.6, physical - 0.15)
    double_preserved = double_talk >= max(0.55, aec_near - 0.2)
    return {
        "near_end_preserved": near_preserved,
        "double_talk_preserved": double_preserved,
        "preliminary_double_talk_pass": near_preserved and double_preserved,
        "product_aec_enabled": False,
    }


def run(*, duration: float, volume_percent: int, confirm=input) -> dict:
    if not 3.0 <= duration <= 8.0:
        raise DoubleTalkError("duration must be between 3 and 8 seconds")
    if not 1 <= volume_percent <= 30:
        raise DoubleTalkError("volume must be between 1 and 30 percent")
    streams = audio_aec_probe._active_streams()
    if any(streams.values()):
        raise DoubleTalkError("audio graph is busy")
    source, _sink = audio_aec_probe._logical_audio()
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    lock_path = runtime_dir / "vq-aec-probe.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    service = EchoCancelService()
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DoubleTalkError("another audio acceptance probe is running") from exc
        with tempfile.TemporaryDirectory(prefix="vq-double-talk-", dir=runtime_dir) as temporary:
            root = Path(temporary)
            stimulus = root / "far-end.wav"
            physical = root / "physical-near.wav"
            aec_near = root / "aec-near.wav"
            double_talk = root / "aec-double.wav"
            audio_aec_probe._write_stimulus(stimulus, duration)
            confirm(f"准备物理麦克风近讲测试。按回车后朗读：{REFERENCE_TEXT}")
            _record(source["endpoint"], physical, duration)
            physical_text = _transcribe(physical)
            try:
                service.start()
                confirm(f"准备 AEC 近讲测试。按回车后重复朗读：{REFERENCE_TEXT}")
                _record(AEC_SOURCE_NAME, aec_near, duration)
                aec_text = _transcribe(aec_near)
                confirm("准备 AEC 双讲测试。按回车后在测试音播放期间重复朗读同一句。")
                _record(
                    AEC_SOURCE_NAME, double_talk, duration,
                    sink=AEC_SINK_NAME, stimulus=stimulus, volume_percent=volume_percent,
                )
                double_text = _transcribe(double_talk)
            finally:
                service.stop()
    scores = {
        "physical_near": _similarity(physical_text),
        "aec_near": _similarity(aec_text),
        "aec_double_talk": _similarity(double_text),
    }
    return {
        "reference_text": REFERENCE_TEXT,
        "transcripts": {
            "physical_near": physical_text,
            "aec_near": aec_text,
            "aec_double_talk": double_text,
        },
        "similarity": scores,
        "decision": _decision(
            scores["physical_near"], scores["aec_near"], scores["aec_double_talk"]
        ),
        "duration_seconds": duration,
        "far_end_volume_percent": volume_percent,
        "artifacts_retained": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-private-speech", action="store_true")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--volume", type=int, default=15)
    arguments = parser.parse_args()
    if not arguments.execute:
        print(json.dumps({
            "ok": True, "status": "dry_run", "reference_text": REFERENCE_TEXT,
            "rounds": ["physical_near", "aec_near", "aec_double_talk"],
            "raw_audio_retained": False, "product_aec_enabled": False,
        }, ensure_ascii=False, indent=2))
        return 0
    if not arguments.acknowledge_private_speech or not sys.stdin.isatty():
        print(json.dumps({
            "ok": False,
            "error": "interactive TTY and --acknowledge-private-speech are required",
        }, ensure_ascii=False))
        return 2
    try:
        result = run(duration=arguments.duration, volume_percent=arguments.volume)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
