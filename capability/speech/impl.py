"""Restricted speech implementation for VQ Runtime."""

from __future__ import annotations

import os
import re
import secrets
import time
from pathlib import Path

from gateway.speech import SpeechEngine, SpeechError
from gateway.speech.tempo import slow_wave, validate_speed
from gateway.speech.qnn_process import QnnTtsProcess
from media_service import MediaClient, MediaClientError
from runtime.errors import HardwareError, ResourceNotFoundError, SchemaValidationError

_MEDIA_ID = re.compile(r"^audio_[A-Za-z0-9_-]{20,64}$")
_MEDIA_OWNER = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_AI_ROOT = Path.home() / ".local/share/vq-ai"
_TTS_LAB_ROOT = Path.home() / ".local/share/vq-tts-lab"
_RUNTIME_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
_MEDIA_ROOT = _RUNTIME_DIR / "vq-phone-gateway/media"
_QNN_ROOT = Path(os.environ.get("VQ_MELO_QNN_ROOT", _AI_ROOT / "tts-qnn"))
_QNN_MODELS = Path(os.environ.get("VQ_MELO_QNN_MODELS", _AI_ROOT / "models/tts/melo-qnn"))
_ENGINE = SpeechEngine(
    whisper=_AI_ROOT / "src/whisper.cpp/build/bin/whisper-cli",
    whisper_model=_AI_ROOT / "models/stt/ggml-base.bin",
    piper_python=_AI_ROOT / "venv/bin/python",
    piper_model=_AI_ROOT / "models/tts/zh_CN-huayan-medium.onnx",
    tts_backend=os.environ.get("VQ_TTS_BACKEND", "piper"),
    matcha_python=_TTS_LAB_ROOT / "venv/bin/python",
    matcha_model_dir=_AI_ROOT / "models/tts/matcha-icefall-zh-baker",
    matcha_vocoder=_AI_ROOT / "models/tts/vocos-22khz-univ.onnx",
    melo_python=_TTS_LAB_ROOT / "venv/bin/python",
    melo_model_dir=_AI_ROOT / "models/tts/vits-melo-tts-zh_en",
    qnn_tts=QnnTtsProcess(_QNN_ROOT / "frontend-venv/bin/python", _QNN_ROOT,
                         _QNN_MODELS, _MEDIA_ROOT,
                         shared_root=os.environ.get("VQ_MELO_QNN_SHARED_ROOT")),
    timeout=29,
)


def _audio_source(media_id: str) -> Path:
    if not _MEDIA_ID.fullmatch(media_id):
        raise SchemaValidationError("media_id格式无效", capability="speech.transcribe")
    matches = list(_MEDIA_ROOT.glob(f"{media_id}.*"))
    if len(matches) != 1 or matches[0].parent.resolve() != _MEDIA_ROOT.resolve():
        raise ResourceNotFoundError("录音不存在或已经过期", capability="speech.transcribe")
    return matches[0]


def _media_client():
    return MediaClient()


def _shared_source(media_id: str, owner: str) -> Path:
    if not _MEDIA_ID.fullmatch(media_id) or not _MEDIA_OWNER.fullmatch(owner):
        raise SchemaValidationError("共享录音引用格式无效", capability="speech.transcribe")
    try:
        value = _media_client().call("resolve", owner=owner, media_id=media_id)
        return Path(value["path"])
    except (OSError, KeyError, MediaClientError) as exc:
        raise ResourceNotFoundError(
            "共享录音不存在或已经过期", capability="speech.transcribe") from exc


def _cleanup_generated() -> None:
    cutoff = time.time() - 600
    for path in _MEDIA_ROOT.glob("runtime-tts-*.wav"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def transcribe(args):
    owner = args.get("media_owner")
    source = (
        _shared_source(args["media_id"], owner)
        if owner is not None else _audio_source(args["media_id"])
    )
    try:
        text = _ENGINE.transcribe(source, _MEDIA_ROOT)
    except SpeechError as exc:
        raise HardwareError(str(exc), capability="speech.transcribe") from exc
    return {"text": text}


def synthesize(args):
    try:
        speed = validate_speed(args.get("speed", 1.0))
    except ValueError as exc:
        raise SchemaValidationError(str(exc), capability="speech.synthesize") from exc
    text = args["text"].strip()
    if not text or len(text) > 2000:
        raise SchemaValidationError("朗读文字长度必须为1到2000个字符", capability="speech.synthesize")
    owner = args.get("media_owner")
    if owner is not None and not _MEDIA_OWNER.fullmatch(owner):
        raise SchemaValidationError("共享音频所有者格式无效", capability="speech.synthesize")
    _MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
    _MEDIA_ROOT.chmod(0o700)
    _cleanup_generated()
    token = secrets.token_urlsafe(18)
    output = _MEDIA_ROOT / f"runtime-tts-{token}.wav"
    try:
        _ENGINE.synthesize(text, output)
        if speed != 1.0:
            try:
                slow_wave(output, speed)
            except Exception as exc:
                raise SpeechError("慢速音频处理失败；不会悄悄改用正常语速") from exc
        os.chmod(output, 0o600)
    except SpeechError as exc:
        output.unlink(missing_ok=True)
        raise HardwareError(str(exc), capability="speech.synthesize") from exc
    if owner is not None:
        try:
            value = _media_client().create_from_path(
                owner, "audio/wav", output, kind="generated"
            )
        except (OSError, MediaClientError) as exc:
            raise HardwareError("Media Service暂时不可用", capability="speech.synthesize") from exc
        finally:
            output.unlink(missing_ok=True)
        return value
    return {"audio_token": token, "mime_type": "audio/wav", "size": output.stat().st_size}


def status(args=None):
    bilingual = _ENGINE.bilingual_tts_ready()
    result = {
        "ready": _ENGINE.ready(),
        "stt": _ENGINE.whisper.is_file() and _ENGINE.whisper_model.is_file(),
        "tts": _ENGINE.tts_ready(),
        "tts_languages": ["zh-CN", "en-US"] if bilingual else ["zh-CN"],
        "tts_bilingual": bilingual,
        "tts_backends": {
            "default": _ENGINE.active_tts_backend(),
            "mixed_zh_en": "melo-qnn-htp" if getattr(_ENGINE, "tts_backend", None) == "melo-qnn" and bilingual else "melo-zh-en" if bilingual else None,
        },
    }
    if getattr(_ENGINE, "tts_backend", None) == "melo-qnn":
        result["npu_worker"] = _ENGINE.qnn_tts.status()
    return result


def close():
    _ENGINE.close()


ACTIONS = {
    "speech.status": status,
    "speech.transcribe": transcribe,
    "speech.synthesize": synthesize,
}
