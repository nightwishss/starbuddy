#!/usr/bin/env python3
"""Speak one bounded NPU evaluation cue through the logical board speaker."""

import argparse
import time

from runtime.client import RuntimeClient
from sdk.audio import AudioClient, AudioError
from sdk.speech import SpeechClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    args = parser.parse_args()
    if not 1 <= len(args.text) <= 100:
        parser.error("cue must contain 1..100 characters")
    owner = "session:pose-npu-eval"
    runtime = RuntimeClient(timeout=35)
    speech, audio = SpeechClient(runtime), AudioClient(runtime)
    media = speech.synthesize(args.text, owner)
    playback = audio.start_playback(media["media_id"], owner, volume_percent=85)
    playback_id = playback["playback_id"]
    state = "pending"
    for _ in range(150):
        state = audio.playback_status(playback_id).get("state", "failed")
        if state in {"completed", "failed", "stopped"}:
            break
        time.sleep(.2)
    if state not in {"completed", "stopped"}:
        try:
            audio.stop_playback(playback_id)
        except AudioError:
            pass
        raise SystemExit(f"cue playback failed: {state}")


if __name__ == "__main__":
    main()
