#!/usr/bin/env python3
"""Private JSONL worker that keeps the local Matcha TTS models resident."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from array import array
from pathlib import Path

import sherpa_onnx


def _load(model_dir: Path, vocoder: Path):
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=str(model_dir / "model-steps-3.onnx"),
                vocoder=str(vocoder),
                lexicon=str(model_dir / "lexicon.txt"),
                tokens=str(model_dir / "tokens.txt"),
            ),
            num_threads=2,
            debug=False,
        ),
        rule_fsts=",".join(
            str(model_dir / name) for name in ("phone.fst", "date.fst", "number.fst")
        ),
        max_num_sentences=1,
    )
    if not config.validate():
        raise ValueError("invalid Matcha model configuration")
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
    if not audio.samples or audio.sample_rate <= 0:
        raise RuntimeError("empty audio")
    pcm = array(
        "h",
        (max(-32768, min(32767, round(sample * 32767))) for sample in audio.samples),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav_file:
        wav_file.setframerate(audio.sample_rate)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
        wav_file.writeframes(pcm.tobytes())
    return {
        "sample_rate": audio.sample_rate,
        "duration_ms": round(len(audio.samples) * 1000 / audio.sample_rate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vocoder", type=Path, required=True)
    args = parser.parse_args()
    tts = _load(args.model_dir, args.vocoder)
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
