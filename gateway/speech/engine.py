"""Local STT and TTS adapters. The phone only transports raw media."""

from __future__ import annotations

import json
import math
import select
import subprocess
import tempfile
import re
import threading
import wave
from pathlib import Path


class SpeechError(RuntimeError):
    pass


def prepare_tts_text(text: str) -> str:
    """Turn display-oriented model output into one natural spoken passage."""
    clean = text.strip()
    clean = re.sub(r"```.*?```", "", clean, flags=re.DOTALL)
    clean = re.sub(r"[`*_#>]", "", clean)
    clean = re.sub(r"(?m)^\s*(?:[-+•]|\d+[.)、])\s*", "", clean)
    clean = re.sub(r"\s*[：:；;]\s*", "，", clean)
    clean = re.sub(r"\s*\n+\s*", "，", clean)
    clean = re.sub(r"[，,]{2,}", "，", clean)
    clean = re.sub(r"[。.!！]{2,}", "。", clean)
    clean = re.sub(r"[？?]{2,}", "？", clean)
    clean = re.sub(r"\s+", " ", clean).strip(" ，")
    # Keep English names intact while giving the bilingual tokenizer an
    # explicit pronunciation boundary between Latin and CJK scripts.
    clean = re.sub(r"([\u3400-\u9fff])([A-Za-z])", r"\1 \2", clean)
    clean = re.sub(r"([A-Za-z])([\u3400-\u9fff])", r"\1 \2", clean)
    return clean


class SpeechEngine:
    def __init__(
        self,
        whisper: Path,
        whisper_model: Path,
        piper_python: Path,
        piper_model: Path,
        *,
        ffmpeg: Path = Path("/usr/bin/ffmpeg"),
        timeout: float = 90.0,
        threads: int = 8,
        length_scale: float = 0.92,
        sentence_silence: float = 0.14,
        persistent_tts: bool = True,
        tts_backend: str = "piper",
        matcha_python: Path | None = None,
        matcha_model_dir: Path | None = None,
        matcha_vocoder: Path | None = None,
        melo_python: Path | None = None,
        melo_model_dir: Path | None = None,
        qnn_tts=None,
    ) -> None:
        self.whisper = whisper
        self.whisper_model = whisper_model
        self.piper_python = piper_python
        self.piper_model = piper_model
        self.ffmpeg = ffmpeg
        self.timeout = timeout
        self.threads = threads
        self.length_scale = length_scale
        self.sentence_silence = sentence_silence
        self.persistent_tts = persistent_tts
        if tts_backend not in {"piper", "matcha", "melo", "melo-qnn"}:
            raise ValueError("unsupported tts_backend")
        self.tts_backend = tts_backend
        self.matcha_python = matcha_python
        self.matcha_model_dir = matcha_model_dir
        self.matcha_vocoder = matcha_vocoder
        self.melo_python = melo_python
        self.melo_model_dir = melo_model_dir
        self._piper_lock = threading.Lock()
        self._piper_process = None
        self._matcha_lock = threading.Lock()
        self._matcha_process = None
        self._melo_lock = threading.Lock()
        self._melo_process = None
        self.qnn_tts = qnn_tts

    def ready(self) -> bool:
        return all(path.is_file() for path in (
            self.whisper, self.whisper_model, self.ffmpeg,
        )) and self.tts_ready()

    def tts_ready(self) -> bool:
        if self.tts_backend == "melo-qnn":
            return self.qnn_tts is not None and self.qnn_tts.configured()
        piper_ready = self.piper_python.is_file() and self.piper_model.is_file()
        if self.tts_backend == "piper":
            return piper_ready
        if self.tts_backend == "melo":
            return self._melo_ready()
        return self._matcha_ready() or piper_ready

    def active_tts_backend(self) -> str:
        if self.tts_backend == "melo-qnn":
            return "melo-qnn-htp" if self.tts_ready() else "unavailable"
        if self.tts_backend == "melo":
            return "melo-zh-en" if self._melo_ready() else "unavailable"
        if self.tts_backend == "matcha" and self._matcha_ready():
            return "matcha"
        return "piper"

    def _matcha_ready(self) -> bool:
        if not all((self.matcha_python, self.matcha_model_dir, self.matcha_vocoder)):
            return False
        return (
            self.matcha_python.is_file()
            and (self.matcha_model_dir / "model-steps-3.onnx").is_file()
            and (self.matcha_model_dir / "lexicon.txt").is_file()
            and (self.matcha_model_dir / "tokens.txt").is_file()
            and self.matcha_vocoder.is_file()
        )

    def _melo_ready(self) -> bool:
        if not all((self.melo_python, self.melo_model_dir)):
            return False
        return (
            self.melo_python.is_file()
            and (self.melo_model_dir / "model.onnx").is_file()
            and (self.melo_model_dir / "lexicon.txt").is_file()
            and (self.melo_model_dir / "tokens.txt").is_file()
        )

    def bilingual_tts_ready(self) -> bool:
        if self.tts_backend == "melo-qnn":
            return self.tts_ready()
        return self._melo_ready()

    def tts_backend_for(self, text: str) -> str:
        """Select one voice for the complete utterance to preserve mixed-language flow."""
        if self.tts_backend == "melo-qnn":
            if not self.tts_ready(): raise SpeechError("NPU 中英文语音模型尚未就绪。")
            return "melo-qnn-htp"
        if self.tts_backend == "melo":
            if not self._melo_ready():
                raise SpeechError("中英文统一语音模型尚未就绪。")
            return "melo-zh-en"
        if re.search(r"[A-Za-z]", text):
            if not self._melo_ready():
                raise SpeechError("英文语音模型尚未就绪。")
            return "melo-zh-en"
        return self.active_tts_backend()

    def transcribe(self, source: Path, work_dir: Path) -> str:
        if not self.whisper.is_file() or not self.whisper_model.is_file():
            raise SpeechError("板端语音模型尚未就绪。")
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=work_dir, prefix="stt-") as temp_name:
                temp = Path(temp_name)
                # Physical turn capture already produces Whisper's input format.
                # Avoid spawning ffmpeg and copying the recording on every turn.
                wav = source if self._whisper_wav_ready(source) else temp / "input.wav"
                output = temp / "transcript"
                if wav != source:
                    self._run(
                        [
                            str(self.ffmpeg), "-nostdin", "-v", "error", "-y", "-i", str(source),
                            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav),
                        ],
                        "无法解码手机录音。",
                    )
                self._run(
                    [
                        str(self.whisper), "-m", str(self.whisper_model), "-f", str(wav),
                        "-l", "zh", "-t", str(min(4, self.threads)),
                        "-ac", str(self._whisper_audio_context(wav)),
                        "-nt", "-np", "-otxt", "-of", str(output),
                    ],
                    "本地语音识别失败。",
                )
                transcript = output.with_suffix(".txt").read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SpeechError("本地语音服务文件访问失败。") from exc
        if not transcript:
            raise SpeechError("没有识别到清晰语音，请靠近麦克风再试。")
        return transcript

    @staticmethod
    def _whisper_audio_context(source: Path) -> int:
        """Cover the complete short WAV plus two seconds, not a fixed crop.

        Whisper's encoder has 50 context positions per audio second. Round
        upwards for its compute blocks and retain at least 10.24 seconds.
        Long/unknown inputs keep the model's full 30-second sliding window (0).
        This does not change audio samples, model weights or decoder search.
        """
        try:
            with wave.open(str(source), "rb") as stream:
                if (stream.getcomptype() != "NONE" or stream.getnchannels() != 1
                        or stream.getsampwidth() != 2 or stream.getframerate() != 16000
                        or stream.getnframes() <= 0):
                    return 0
                duration = stream.getnframes() / stream.getframerate()
            context = max(512, math.ceil((duration + 2) * 50 / 64) * 64)
            return context if context < 1500 else 0
        except (OSError, EOFError, wave.Error):
            return 0

    @staticmethod
    def _whisper_wav_ready(source: Path) -> bool:
        try:
            with wave.open(str(source), "rb") as stream:
                return (stream.getcomptype() == "NONE" and stream.getnchannels() == 1
                        and stream.getsampwidth() == 2 and stream.getframerate() == 16000
                        and stream.getnframes() > 0)
        except (OSError, EOFError, wave.Error):
            return False

    def synthesize(self, text: str, output: Path) -> None:
        clean = prepare_tts_text(text)
        if not clean:
            raise SpeechError("没有可合成的回复。")
        if len(clean) > 2000:
            clean = clean[:2000] + "。回复过长，已截断朗读。"
        output.parent.mkdir(parents=True, exist_ok=True)
        backend = self.tts_backend_for(clean)
        if backend == "melo-qnn-htp":
            from gateway.speech.qnn_process import QnnSpeechError
            try:
                self.qnn_tts.synthesize(clean, output)
            except QnnSpeechError as exc:
                raise SpeechError(str(exc)) from exc
        elif backend == "melo-zh-en":
            self._run_melo_worker(clean, output)
        elif backend == "matcha":
            try:
                self._run_matcha_worker(clean, output)
            except SpeechError:
                self._run_piper_worker(clean, output)
        elif self.persistent_tts:
            self._run_piper_worker(clean, output)
        else:
            self._run(
                [
                    str(self.piper_python), "-m", "piper", "-m", str(self.piper_model),
                    "-f", str(output),
                    "--length-scale", str(self.length_scale),
                    "--sentence-silence", str(self.sentence_silence),
                    "--", clean,
                ],
                "本地语音合成失败。",
            )
        if not output.is_file() or output.stat().st_size == 0:
            raise SpeechError("本地语音合成没有产生音频。")

    def _run_piper_worker(self, text: str, output: Path) -> None:
        request = json.dumps({
            "text": text, "output": str(output),
            "length_scale": self.length_scale,
            "sentence_silence": self.sentence_silence,
        }, ensure_ascii=False, separators=(",", ":"))
        with self._piper_lock:
            try:
                process = self._ensure_piper_worker()
                process.stdin.write(request + "\n")
                process.stdin.flush()
                ready, _, _ = select.select([process.stdout], [], [], self.timeout)
                if not ready:
                    raise TimeoutError
                line = process.stdout.readline()
                response = json.loads(line)
                if not response.get("ok"):
                    raise RuntimeError
            except (OSError, BrokenPipeError, TimeoutError, RuntimeError,
                    ValueError, json.JSONDecodeError) as exc:
                self._stop_piper_worker()
                raise SpeechError("本地语音合成失败。") from exc

    def _ensure_piper_worker(self):
        process = self._piper_process
        if process is not None and process.poll() is None:
            return process
        self._piper_process = subprocess.Popen(
            [
                str(self.piper_python), "-m", "gateway.speech.piper_worker",
                "--model", str(self.piper_model),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
        )
        return self._piper_process

    def _stop_piper_worker(self) -> None:
        process = self._piper_process
        self._piper_process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)

    def _run_matcha_worker(self, text: str, output: Path) -> None:
        request = json.dumps({
            "text": text, "output": str(output), "speed": 1.0,
        }, ensure_ascii=False, separators=(",", ":"))
        with self._matcha_lock:
            try:
                process = self._ensure_matcha_worker()
                process.stdin.write(request + "\n")
                process.stdin.flush()
                ready, _, _ = select.select([process.stdout], [], [], self.timeout)
                if not ready:
                    raise TimeoutError
                response = json.loads(process.stdout.readline())
                if not response.get("ok"):
                    raise RuntimeError
            except (OSError, BrokenPipeError, TimeoutError, RuntimeError,
                    ValueError, json.JSONDecodeError) as exc:
                self._stop_matcha_worker()
                raise SpeechError("Matcha语音合成失败。") from exc

    def _ensure_matcha_worker(self):
        process = self._matcha_process
        if process is not None and process.poll() is None:
            return process
        self._matcha_process = subprocess.Popen(
            [
                str(self.matcha_python), "-m", "gateway.speech.matcha_worker",
                "--model-dir", str(self.matcha_model_dir),
                "--vocoder", str(self.matcha_vocoder),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
        )
        return self._matcha_process

    def _stop_matcha_worker(self) -> None:
        process = self._matcha_process
        self._matcha_process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)

    def _run_melo_worker(self, text: str, output: Path) -> None:
        request = json.dumps({
            "text": text, "output": str(output), "speed": 1.0,
        }, ensure_ascii=False, separators=(",", ":"))
        with self._melo_lock:
            try:
                process = self._ensure_melo_worker()
                process.stdin.write(request + "\n")
                process.stdin.flush()
                ready, _, _ = select.select([process.stdout], [], [], self.timeout)
                if not ready:
                    raise TimeoutError
                response = json.loads(process.stdout.readline())
                if not response.get("ok"):
                    raise RuntimeError
            except (OSError, BrokenPipeError, TimeoutError, RuntimeError,
                    ValueError, json.JSONDecodeError) as exc:
                self._stop_melo_worker()
                raise SpeechError("中英文语音合成失败。") from exc

    def _ensure_melo_worker(self):
        process = self._melo_process
        if process is not None and process.poll() is None:
            return process
        self._melo_process = subprocess.Popen(
            [
                str(self.melo_python), "-m", "gateway.speech.melo_worker",
                "--model-dir", str(self.melo_model_dir),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
        )
        return self._melo_process

    def _stop_melo_worker(self) -> None:
        process = self._melo_process
        self._melo_process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)

    def close(self):
        if self.qnn_tts is not None:
            self.qnn_tts.close()
        self._stop_melo_worker()
        self._stop_matcha_worker()
        self._stop_piper_worker()

    def _run(self, command: list[str], public_error: str) -> None:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SpeechError(public_error) from exc
        if result.returncode != 0:
            raise SpeechError(public_error)
