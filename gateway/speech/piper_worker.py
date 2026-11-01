#!/usr/bin/env python3
"""Private JSONL worker that keeps a Piper voice loaded between requests."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

from piper.config import SynthesisConfig
from piper.voice import PiperVoice


def _synthesize(voice, request: dict) -> None:
    text = request.get("text")
    output = Path(str(request.get("output", "")))
    if (not isinstance(text, str) or not text or len(text) > 2020
            or not output.is_absolute() or output.suffix != ".wav"
            or not output.name.startswith("runtime-tts-")):
        raise ValueError("invalid request")
    length_scale = float(request.get("length_scale", 0.92))
    sentence_silence = float(request.get("sentence_silence", 0.08))
    if not 0.5 <= length_scale <= 2.0 or not 0 <= sentence_silence <= 2.0:
        raise ValueError("invalid synthesis controls")
    config = SynthesisConfig(length_scale=length_scale)
    silence = bytes(int(voice.config.sample_rate * sentence_silence * 2))
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav_file:
        configured = False
        for index, chunk in enumerate(voice.synthesize(text, config)):
            if not configured:
                wav_file.setframerate(chunk.sample_rate)
                wav_file.setsampwidth(chunk.sample_width)
                wav_file.setnchannels(chunk.sample_channels)
                configured = True
            if index:
                wav_file.writeframes(silence)
            wav_file.writeframes(chunk.audio_int16_bytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    voice = PiperVoice.load(Path(args.model))
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("invalid request")
            _synthesize(voice, request)
            response = {"ok": True}
        except Exception:
            response = {"ok": False}
        print(json.dumps(response, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
