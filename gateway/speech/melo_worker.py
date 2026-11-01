#!/usr/bin/env python3
"""Private JSONL worker for the local Chinese/English MeloTTS voice."""

from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from array import array
from pathlib import Path

import sherpa_onnx


def _melo_threads() -> int:
    value = os.environ.get("VQ_MELO_THREADS", "2")
    if value not in ("1", "2", "3", "4"):
        raise ValueError("VQ_MELO_THREADS must be an integer from 1 to 4")
    return int(value)


def _load(model_dir: Path):
    num_threads = _melo_threads()
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(model_dir / "model.onnx"),
                lexicon=str(model_dir / "lexicon.txt"),
                tokens=str(model_dir / "tokens.txt"),
                length_scale=1.0,
            ),
            num_threads=num_threads,
            debug=False,
        ),
        rule_fsts=",".join(
            str(model_dir / name) for name in ("phone.fst", "date.fst", "number.fst")
        ),
        max_num_sentences=1,
    )
    if not config.validate():
        raise ValueError("invalid MeloTTS model configuration")
    return sherpa_onnx.OfflineTts(config)


def _synthesize(tts, request: dict) -> dict:
    text = request.get("text")
    output = Path(str(request.get("output", "")))
    speed = float(request.get("speed", 1.0))
    if (
        not isinstance(text, str) or not text or len(text) > 2020
        or not output.is_absolute() or output.suffix != ".wav"
        or not output.name.startswith("runtime-tts-")
        or not 0.8 <= speed <= 1.2
    ):
        raise ValueError("invalid request")
    audio = tts.generate(text, sid=0, speed=speed)
    samples = audio.samples
    if len(samples) == 0 or audio.sample_rate <= 0:
        raise RuntimeError("empty audio")
    output.parent.mkdir(parents=True, exist_ok=True)
    native_write = getattr(sherpa_onnx, "write_wave", None)
    if native_write is not None:
        # The installed sherpa-onnx already provides a C++ PCM WAV writer;
        # avoid a Python loop per sample without adding another dependency.
        if not native_write(str(output), samples, audio.sample_rate):
            raise RuntimeError("WAV write failed")
    else:
        pcm = array("h", (max(-32768, min(32767, round(sample * 32767))) for sample in samples))
        if sys.byteorder != "little":
            pcm.byteswap()
        with wave.open(str(output), "wb") as wav_file:
            wav_file.setframerate(audio.sample_rate)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)
            wav_file.writeframes(pcm.tobytes())
    return {
        "sample_rate": audio.sample_rate,
        "duration_ms": round(len(samples) * 1000 / audio.sample_rate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    tts = _load(args.model_dir)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("invalid request")
            response = {"ok": True, **_synthesize(tts, request)}
        except Exception:
            response = {"ok": False}
        print(json.dumps(response, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
