#!/usr/bin/env python3
"""Repeat and concurrently exercise board-local speech Runtime capabilities."""

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.client import RuntimeClient


def max_temperature_c():
    values = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            values.append(float(path.read_text().strip()) / 1000)
        except (OSError, ValueError):
            pass
    return max(values) if values else None


def require_ok(response, capability):
    if not response.get("ok"):
        raise RuntimeError(f"{capability} failed: {response.get('error', {}).get('code')}")
    return response


def synthesize_once(index):
    client = RuntimeClient(timeout=35)
    response = require_ok(
        client.call(
            "speech.synthesize",
            {"text": f"并发语音能力测试第{index + 1}路。"},
            timeout_ms=30_000,
        ),
        "speech.synthesize",
    )
    token = response["result"]["audio_token"]
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    output = runtime_dir / "vq-phone-gateway/media" / f"runtime-tts-{token}.wav"
    output.unlink(missing_ok=True)
    return response["meta"]["elapsed_ms"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media_id")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.cycles <= 20 or not 1 <= args.concurrency <= 8:
        parser.error("cycles must be 1..20 and concurrency 1..8")

    before = max_temperature_c()
    stt_ms = []
    transcripts = []
    client = RuntimeClient(timeout=35)
    for _ in range(args.cycles):
        response = require_ok(
            client.call(
                "speech.transcribe", {"media_id": args.media_id}, timeout_ms=30_000
            ),
            "speech.transcribe",
        )
        stt_ms.append(response["meta"]["elapsed_ms"])
        transcripts.append(response["result"]["text"])

    concurrent_started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        tts_ms = list(pool.map(synthesize_once, range(args.concurrency)))
    concurrent_wall_ms = round((time.monotonic() - concurrent_started) * 1000, 2)

    report = {
        "cycles": args.cycles,
        "transcripts_consistent": len(set(transcripts)) == 1,
        "transcript": transcripts[0],
        "stt_ms": stt_ms,
        "stt_mean_ms": round(statistics.mean(stt_ms), 2),
        "stt_max_ms": max(stt_ms),
        "tts_concurrency": args.concurrency,
        "tts_runtime_elapsed_ms": tts_ms,
        "tts_concurrent_wall_ms": concurrent_wall_ms,
        "temperature_before_c": before,
        "temperature_after_c": max_temperature_c(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
