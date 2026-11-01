"""Audio Capability 实现 — Stream 模型.

stream 生命周期 (模块级 stream_id -> stream 对象):
  audio.open_playback(path)          -> PlaybackStream, 返回 stream_id + 流信息
  audio.play(stream_id, frames)      -> 读 PCM chunk (返回 samples/position/eof)
  audio.close_playback(stream_id)    -> 关闭并移除
  audio.open_capture(path, sr, ch, sw) -> CaptureStream
  audio.write(stream_id, samples)    -> 写 PCM chunk
  audio.close_capture(stream_id)     -> 关闭 (finalize WAV)

资源注册表同 display: 放在 capability 内部 (模块级 dict), 不侵入共享 Runtime。
"""
import subprocess
import uuid

from runtime.errors import ResourceNotFoundError
from .backend import PlaybackStream, CaptureStream
from .physical import (
    capture_clip, capture_turn, interaction_status, playback_status,
    start_playback, stop_playback,
    start_stream_playback, append_stream_playback, finish_stream_playback,
)

_STREAMS = {}  # stream_id -> PlaybackStream | CaptureStream


def _get(stream_id):
    s = _STREAMS.get(stream_id)
    if s is None:
        raise ResourceNotFoundError(
            f"audio stream 不存在或已关闭: {stream_id}",
            capability="audio", details={"stream_id": stream_id})
    return s


def list_devices(args=None):
    cards = []
    try:
        with open("/proc/asound/cards") as f:
            for line in f:
                line = line.strip()
                if line and line[0].isdigit() and " - " in line:
                    cards.append(line.split(" - ", 1)[1])
    except OSError:
        pass
    return {"backends": ["file"], "devices": cards, "count": len(cards)}


def get_info(args=None):
    server = "unknown"
    try:
        r = subprocess.run(["pactl", "info"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.splitlines():
            if line.startswith("Server Name"):
                server = line.split(":", 1)[1].strip()
    except Exception:
        pass
    return {
        "backend": "file", "server": server, "formats": ["wav"],
        "loopback": "snd-aloop available (=m, not loaded)",
    }


def open_playback(args):
    s = PlaybackStream(args["path"])
    stream_id = uuid.uuid4().hex[:8]
    _STREAMS[stream_id] = s
    info = s.info()
    info["stream_id"] = stream_id
    return info


def play(args):
    s = _get(args["stream_id"])
    r = s.read(args.get("frames", 1024))
    r["stream_id"] = args["stream_id"]
    return r


def close_playback(args):
    s = _get(args["stream_id"])
    s.close()
    del _STREAMS[args["stream_id"]]
    return {"stream_id": args["stream_id"], "closed": True}


def open_capture(args):
    s = CaptureStream(
        args["path"], args.get("sample_rate", 16000),
        args.get("channels", 1), args.get("sample_width", 2))
    stream_id = uuid.uuid4().hex[:8]
    _STREAMS[stream_id] = s
    return {
        "stream_id": stream_id, "path": args["path"],
        "sample_rate": s.sample_rate, "channels": s.channels,
        "sample_width": s.sample_width,
    }


def write(args):
    s = _get(args["stream_id"])
    r = s.write(args["samples"])
    r["stream_id"] = args["stream_id"]
    return r


def close_capture(args):
    s = _get(args["stream_id"])
    s.close()
    del _STREAMS[args["stream_id"]]
    return {"stream_id": args["stream_id"], "closed": True, "path": s.path}


# action 名 -> 实现函数 (由 capability/audio/__init__.py 注册进 Registry)
ACTIONS = {
    "audio.list_devices": list_devices,
    "audio.get_info": get_info,
    "audio.open_playback": open_playback,
    "audio.play": play,
    "audio.close_playback": close_playback,
    "audio.open_capture": open_capture,
    "audio.write": write,
    "audio.close_capture": close_capture,
    "audio.capture_clip": capture_clip,
    "audio.capture_turn": capture_turn,
    "audio.start_playback": start_playback,
    "audio.start_stream_playback": start_stream_playback,
    "audio.append_stream_playback": append_stream_playback,
    "audio.finish_stream_playback": finish_stream_playback,
    "audio.playback_status": playback_status,
    "audio.stop_playback": stop_playback,
    "audio.interaction_status": interaction_status,
}
